"""
TensorRT inference wrapper for quantized VGGT.

Loads a serialized .trt engine, manages CUDA I/O buffers, and provides a
simple forward() interface compatible with the existing VGGT evaluation code.

Usage:
  from deploy.runtime.trt_inference import TRTInferenceEngine

  engine = TRTInferenceEngine('deploy/engines/vggt_w8a8.trt')
  output = engine.infer({'images': images_tensor})  # images: (1, N, 3, H, W)
  engine.close()
"""

import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import torch.nn as nn


class TRTInferenceEngine:
    """
    Minimal TensorRT inference engine wrapper.

    Manages:
    - Engine deserialization
    - CUDA memory allocation (input + output buffers)
    - Binding shape updates for dynamic axes (frames dimension)
    - Synchronous inference execution
    """

    def __init__(self, engine_path: str, device_index: int = 0,
                 defer_context: bool = False):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            # Retain the device's PRIMARY context (the one PyTorch also uses) so
            # TRT and torch share one CUDA context — avoids the dual-context
            # conflicts of pycuda.autoinit when the eval harness uses torch-CUDA.
            try:
                import pycuda.autoprimaryctx  # noqa: F401  (PyCUDA >= 2021.1)
            except ImportError:
                import pycuda.autoinit  # noqa: F401  (fallback: separate context)
        except ImportError as e:
            raise ImportError(f'Missing TRT/PyCUDA dependency: {e}')

        self._trt = trt
        self._cuda = cuda

        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)

        self._check_sidecar(engine_path, trt)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f'Failed to deserialize {engine_path}. TRT engines are not portable '
                f'across GPU architectures or TensorRT versions — rebuild it on THIS '
                f'device (see deploy/export/trt_builder.py).')

        # If the engine was built with --allowWeightStreaming (a BUILD-time memory
        # crutch on the 7.4 GB Orin), keep ALL weights resident at inference — at
        # runtime there's no ~3 GB builder library, so the weights fit and we avoid
        # the per-inference streaming transfers. Budget = full streamable size.
        try:
            sw = getattr(self.engine, 'streamable_weights_size', 0) or 0
            if sw > 0:
                for attr in ('weight_streaming_budget_v2', 'weight_streaming_budget'):
                    if hasattr(self.engine, attr):
                        setattr(self.engine, attr, sw)
                        print(f'Weight-streaming engine: pinned {sw/1024/1024:.0f} MB '
                              f'fully resident (no runtime streaming).')
                        break
        except Exception as e:
            print(f'weight-streaming budget not set ({e}); engine may stream at runtime.')

        self._stream = cuda.Stream()
        self._buffers = {}   # name → (host_buf, device_buf)
        self._output_names = []

        # Identify outputs (all bindings that are not inputs)
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.OUTPUT:
                self._output_names.append(name)

        # Scratch/activation memory this engine's context needs. When several engines
        # run SEQUENTIALLY (the all-TRT chain), a single buffer of the MAX size can be
        # shared across all their contexts instead of each grabbing its own ~0.5 GB —
        # the difference between fitting and OOM on the 7.6 GB Orin.
        dm = getattr(self.engine, 'device_memory_size', 0) or 0
        self.device_memory_size = int(dm)
        self.context = None
        if not defer_context:
            self.create_context()

        print(f'TRT engine loaded: {engine_path}')
        print(f'  Inputs:  {[self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors) if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]}')
        print(f'  Outputs: {self._output_names}')

    def create_context(self, shared_device_mem=None, shared_device_mem_size: int = 0):
        """Create the execution context. If a shared device-memory buffer at least as
        big as this engine's device_memory_size is given, bind it instead of letting
        TRT allocate a private scratch block — so N sequential engines share one
        ~0.5 GB buffer rather than N of them. Otherwise TRT allocates its own."""
        if (shared_device_mem is not None
                and shared_device_mem_size >= self.device_memory_size > 0):
            self.context = self.engine.create_execution_context_without_device_memory()
            try:
                self.context.set_device_memory(int(shared_device_mem))
            except TypeError:                      # newer binding also takes the size
                self.context.set_device_memory(int(shared_device_mem),
                                                int(shared_device_mem_size))
        else:
            self.context = self.engine.create_execution_context()
        return self.context

    def _check_sidecar(self, engine_path: str, trt):
        """If a <engine>.json sidecar exists, refuse to load an engine built for a
        different GPU arch / TRT major.minor — the deserialize would fail anyway, but
        with an opaque error. Missing sidecar → skip (best-effort, back-compat)."""
        sidecar = engine_path + '.json'
        if not os.path.exists(sidecar):
            return
        try:
            with open(sidecar) as f:
                info = json.load(f)
        except (OSError, ValueError):
            return
        # Current device compute capability.
        cur_cc = None
        try:
            if torch.cuda.is_available():
                mj, mn = torch.cuda.get_device_capability(0)
                cur_cc = f'sm_{mj}{mn}'
        except Exception:
            pass
        built_cc = info.get('compute_capability')
        if cur_cc and built_cc and cur_cc != built_cc:
            raise RuntimeError(
                f'Engine {engine_path} was built for {built_cc} but this device is '
                f'{cur_cc}. TRT engines are not portable across GPU architectures — '
                f'rebuild it here with deploy/export/trt_builder.py.')
        built_trt = (info.get('trt_version') or '').split('.')[:2]
        cur_trt = trt.__version__.split('.')[:2]
        if built_trt and built_trt != cur_trt:
            raise RuntimeError(
                f'Engine {engine_path} was built with TensorRT {info.get("trt_version")} '
                f'but the runtime is {trt.__version__} — rebuild it here.')

    def _np_dtype(self, name: str):
        """numpy dtype the engine expects/produces for this binding."""
        return self._trt.nptype(self.engine.get_tensor_dtype(name))

    def _alloc_buffer(self, name: str, shape: tuple, dtype=None):
        import pycuda.driver as cuda
        if dtype is None:
            dtype = self._np_dtype(name)
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        # Pageable host buffer (np.empty) rather than pinned (cuda.pagelocked_empty):
        # on the memory-tight Jetson, pinning I/O buffers for several chained engines
        # exhausts the non-swappable pinned pool (cuMemHostAlloc OOM). Pageable memory
        # can use swap; the H2D/D2H copies are marginally slower but never OOM here.
        host = np.empty(int(np.prod(shape)), dtype=dtype)
        device = cuda.mem_alloc(nbytes)
        self._buffers[name] = (host, device)

    def infer(self, inputs: dict) -> dict:
        """
        Run one inference pass.

        Args:
            inputs: dict mapping input tensor name to torch.Tensor or np.ndarray.
                    Tensors are automatically moved to CPU and converted to float32.

        Returns:
            dict mapping output tensor name to np.ndarray.
        """
        import pycuda.driver as cuda

        for name, tensor in inputs.items():
            in_dtype = self._np_dtype(name)
            if isinstance(tensor, torch.Tensor):
                arr = tensor.detach().cpu().numpy().astype(in_dtype, copy=False)
            else:
                arr = np.ascontiguousarray(tensor, dtype=in_dtype)

            shape = arr.shape
            # Update dynamic shape binding for this input
            self.context.set_input_shape(name, shape)

            # host_buf is 1-D (size == prod(shape)); compare element COUNT, not shape,
            # else this never matches arr's multi-dim shape and we'd re-alloc (and leak
            # the old device buffer) every call — fatal across the 6-engine chain.
            if name not in self._buffers or self._buffers[name][0].size != arr.size:
                self._alloc_buffer(name, shape)

            host_buf, dev_buf = self._buffers[name]
            np.copyto(host_buf, arr.ravel())
            cuda.memcpy_htod_async(dev_buf, host_buf, self._stream)
            self.context.set_tensor_address(name, int(dev_buf))

        # Allocate output buffers based on inferred shapes
        for name in self._output_names:
            out_shape = tuple(self.context.get_tensor_shape(name))
            if name not in self._buffers or self._buffers[name][0].shape[0] != int(np.prod(out_shape)):
                self._alloc_buffer(name, out_shape)
            _, dev_buf = self._buffers[name]
            self.context.set_tensor_address(name, int(dev_buf))

        # Execute
        self.context.execute_async_v3(stream_handle=self._stream.handle)
        self._stream.synchronize()

        # Copy outputs back to host
        outputs = {}
        for name in self._output_names:
            out_shape = tuple(self.context.get_tensor_shape(name))
            host_buf, dev_buf = self._buffers[name]
            cuda.memcpy_dtoh_async(host_buf, dev_buf, self._stream)
            self._stream.synchronize()
            outputs[name] = host_buf.reshape(out_shape).copy()

        return outputs

    def close(self):
        del self.context
        del self.engine

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ──────────────────────────────────────────────────────────────────────────────
# Drop-in nn.Module wrapper for run_evaluation_vggt
# ──────────────────────────────────────────────────────────────────────────────

class TRTPoseModel(nn.Module):
    """Hybrid pose model: PyTorch DINO embed → TRT blocks engine → PyTorch camera head.

    The TRT engine covers ONLY the aggregator's frame/global block stack (the heavy,
    quantized region — see scripts/AggregatorBlocks). DINO patch-embed + token
    assembly (Aggregator.embed_tokens) and the camera head run in PyTorch FP16 around
    it; both are un-quantized and identical across every approach, so the engine is
    exactly the variable under study. The torch block weights are dropped after load
    so we don't double-store ~1.2 GB that already live in the engine.

    Looks like the PyTorch VGGT to the eval harness: forward(images,
    frames_chunk_size=...) → {"pose_enc": (B,S,9)}. `run_evaluation_vggt` times that
    call, so end-to-end latency includes the torch embed/head AND the H2D/D2H token
    transfers (the honest deployment number — only the blocks are accelerated).
    """

    def __init__(self, engine_path: str, checkpoint: str, device: str = None,
                 dtype: torch.dtype = torch.float16, height: int = 350, width: int = 518):
        super().__init__()
        self.engine = TRTInferenceEngine(engine_path)
        self._out_name = self.engine._output_names[0]   # "last_tokens"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        # CO3D preprocessing yields a variable height per aspect ratio; resize eval
        # frames to the fixed (H,W) the engine's token grid was exported for so
        # embed_tokens produces the P the "tokens" binding expects. Same resize for
        # every approach → comparisons between TRT rows stay apples-to-apples.
        self._in_hw = (height, width)
        self._load_torch_parts(checkpoint)

    def _load_torch_parts(self, checkpoint: str):
        """Load DINO embed + camera head in torch FP16; drop the block stack (the
        TRT engine owns it) so the torch side stays light."""
        import gc
        scripts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import eval_co3d_realquant as erq   # proven Jetson FP16 loader
        full = erq._load_model(checkpoint, self.dtype, self.device)
        full.depth_head = full.point_head = full.track_head = None
        self.aggregator = full.aggregator
        self.camera_head = full.camera_head
        # The heavy block weights now live in the TRT engine — free the torch copies
        # (~1.2 GB). KEEP `rope`: embed_tokens needs it to build `pos` (it does
        # `pos = pos + 1`, which crashes if rope is None → pos None); rope is tiny.
        self.aggregator.frame_blocks = None
        self.aggregator.global_blocks = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def forward(self, images: torch.Tensor, frames_chunk_size=None, **kwargs):
        images = images.to(self.device, self.dtype)
        if images.dim() == 4:        # CO3D loader gives (S,C,H,W); embed_tokens needs (B,S,C,H,W)
            images = images.unsqueeze(0)
        if tuple(images.shape[-2:]) != self._in_hw:
            b, s = images.shape[:2]
            flat = images.reshape(b * s, *images.shape[2:])
            flat = torch.nn.functional.interpolate(
                flat, size=self._in_hw, mode="bilinear", align_corners=False)
            images = flat.reshape(b, s, *flat.shape[1:])
        with torch.inference_mode():
            # 1. PyTorch: DINO patch-embed + token assembly.
            tokens, _pos, _B, _S, _P, _C = self.aggregator.embed_tokens(images)
            # 2. TRT: the blocks engine (tokens → last-block tokens).
            last = self.engine.infer({"tokens": tokens})[self._out_name]
            last_t = torch.from_numpy(last).to(device=self.device, dtype=self.dtype)
            # 3. PyTorch: camera head (reads only the last block's tokens).
            pose_enc = self.camera_head([last_t])[-1]
        return {"pose_enc": pose_enc.float()}

    def close(self):
        self.engine.close()


class TRTChainedPoseModel(nn.Module):
    """Hybrid pose model whose transformer is covered by N CHAINED TRT sub-engines.

    The aggregator's 24 attention pairs are split into N contiguous chunks, each its
    own engine (a chunk's fused weight block is ~48 MB × pairs, small enough to build
    within the Orin Nano's contiguous-memory limit — the whole 24-pair stack's ~1.15 GB
    block cannot). DINO embed + camera head run in PyTorch FP16 around the chain, like
    TRTPoseModel. The running token state (B*S, P, C) passes between engines:
    engines[:-1] output 'tokens_out'; engines[-1] outputs the camera-head input
    'last_tokens' [B,S,P,2C].

    Engine paths must be given in pair order (chunk 0 first). Same forward signature as
    TRTPoseModel, so run_evaluation_vggt times the whole chain (embed + N engines +
    inter-engine transfers + camera head) — the honest end-to-end number."""

    def __init__(self, engine_paths, checkpoint: str, device: str = None,
                 dtype: torch.dtype = torch.float16, height: int = 350, width: int = 518):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._in_hw = (height, width)
        # ORDER MATTERS on the 7.6 GB unified board:
        # 1. Load the torch parts FIRST — erq._load_model peaks at ~2.4 GB (full model)
        #    then we drop the block weights down to ~0.7 GB (embed + camera head). Doing
        #    this with the engines NOT yet resident keeps the peak ~2.4 GB instead of
        #    ~3.6 GB (engines + full-model load) which OOMs.
        TRTPoseModel._load_torch_parts(self, checkpoint)   # embed + camera head; drop blocks
        # 2. Defrag NvMap (the dropped blocks freed ~1.7 GB) so the engine deserialize
        #    allocations land in one coalesced region instead of failing on fragmentation.
        self._defrag_nvmap(sum(os.path.getsize(p) for p in engine_paths))
        # 3. Load the engines into the freed/coalesced space (~1.2 GB).
        self.engines = [TRTInferenceEngine(p) for p in engine_paths]
        self._out_names = [e._output_names[0] for e in self.engines]
        print(f"TRT chain: {len(self.engines)} sub-engines "
              f"({[os.path.basename(p) for p in engine_paths]})")

    def _defrag_nvmap(self, engine_bytes):
        """Reserve a contiguous block ~(sum of engine sizes + 1 GB) and free it so the
        engine deserialize allocations land in coalesced NvMap space. Best-effort."""
        if "cuda" not in str(self.device):
            return
        mb = int(engine_bytes / 1024 / 1024) + 1024
        try:
            blk = torch.empty(mb * 1024 * 1024 // 2, dtype=torch.float16, device="cuda")
            del blk
            print(f"NvMap defrag: reserved+freed {mb} MB contiguous before engine load.")
        except Exception as e:
            print(f"NvMap defrag skipped ({e}).")
        finally:
            torch.cuda.empty_cache()

    def forward(self, images: torch.Tensor, frames_chunk_size=None, **kwargs):
        images = images.to(self.device, self.dtype)
        if images.dim() == 4:        # CO3D loader gives (S,C,H,W); embed_tokens needs (B,S,C,H,W)
            images = images.unsqueeze(0)
        if tuple(images.shape[-2:]) != self._in_hw:
            b, s = images.shape[:2]
            flat = images.reshape(b * s, *images.shape[2:])
            flat = torch.nn.functional.interpolate(
                flat, size=self._in_hw, mode="bilinear", align_corners=False)
            images = flat.reshape(b, s, *flat.shape[1:])
        with torch.inference_mode():
            # 1. PyTorch: DINO patch-embed + token assembly → token state (B*S, P, C).
            tokens, _pos, _B, _S, _P, _C = self.aggregator.embed_tokens(images)
            # 2. TRT: run the chunk engines in order; each feeds the next its tokens.
            for eng, out_name in zip(self.engines[:-1], self._out_names[:-1]):
                out = eng.infer({"tokens": tokens})[out_name]
                tokens = torch.from_numpy(out).to(device=self.device, dtype=self.dtype)
            # 3. Last engine emits the camera-head input [B,S,P,2C].
            last = self.engines[-1].infer({"tokens": tokens})[self._out_names[-1]]
            last_t = torch.from_numpy(last).to(device=self.device, dtype=self.dtype)
            # 4. PyTorch: camera head.
            pose_enc = self.camera_head([last_t])[-1]
        return {"pose_enc": pose_enc.float()}

    def close(self):
        for e in self.engines:
            e.close()


class TRTAllEngineModel(nn.Module):
    """Config #3: the ENTIRE pose path in TensorRT via a chain of single-I/O engines,
    with NO PyTorch compute. The monolithic full-model engine cannot build on the Orin
    Nano (DINO's 605 MB weight const + the unplaceable camera-head fusion), so the path
    is split into engines that each build under the contiguous-memory ceiling:

        images → dino_0 → dino_1 → chain_s0 → chain_s8 → chain_s16 → camera_head → pose_enc

    Every engine here has exactly one input and one output, and adjacent shapes already
    match (dino_1's 'tokens' == the aggregator embed state the chain consumes; the last
    chain engine emits the camera-head input [B,S,P,2C]; camera_head emits pose_enc), so
    we just pipe each engine's output array straight into the next engine's sole input.

    Unlike TRTChainedPoseModel there is no torch embed/head — DINO and the camera head
    are themselves engines. No checkpoint is loaded. Engine paths must be in execution
    order (dino_0 first, camera_head last). Engines are fixed-shape (S=10), matching the
    10-frame benchmark; forward resizes frames to the exported (H,W)."""

    def __init__(self, engine_paths, device: str = None,
                 dtype: torch.dtype = torch.float16, height: int = 350, width: int = 518,
                 **_ignored):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._in_hw = (height, width)
        self._shared_dm = None
        # 1. Coalesce NvMap FIRST — each engine's deserialize cudaMallocs its ~0.4 GB
        #    weight block, which needs to be CONTIGUOUS. Reserving+freeing a big torch
        #    block returns coalesced physical memory to the driver so those six weight
        #    allocs (+ the shared scratch) land in one region instead of failing on
        #    fragmentation. MUST happen before any deserialize.
        want_mb = int(sum(os.path.getsize(p) for p in engine_paths) / 1024 / 1024) + 768
        self._defrag(want_mb)
        # 2. Deserialize all engines WITHOUT contexts — weights become resident
        #    (~2.3 GB total) but no per-engine scratch is grabbed yet. Six private
        #    contexts would each reserve ~0.5 GB and OOM the board.
        self.engines = [TRTInferenceEngine(p, defer_context=True) for p in engine_paths]
        # 3. ONE shared scratch buffer sized to the largest engine; every engine runs
        #    strictly sequentially in forward(), so they can all bind the same buffer.
        import pycuda.driver as cuda
        self._dm_size = max(e.device_memory_size for e in self.engines)
        self._shared_dm = cuda.mem_alloc(self._dm_size) if self._dm_size > 0 else None
        for e in self.engines:
            e.create_context(self._shared_dm, self._dm_size)
        self._in_names = [self._sole_input_name(e) for e in self.engines]
        self._out_names = [e._output_names[0] for e in self.engines]
        print(f"TRT all-engine chain: {len(self.engines)} engines, shared scratch "
              f"{self._dm_size/1024/1024:.0f} MB "
              f"({[os.path.basename(p) for p in engine_paths]})")

    def _defrag(self, target_mb):
        """Reserve+free the largest contiguous block we can (down from target_mb) via
        the CUDA DRIVER (pycuda cuMemAlloc), returning it so the engines' contiguous
        weight allocs coalesce. Deliberately NOT torch: a multi-GB torch alloc/free
        leaves torch's CUDACachingAllocator NVML accounting inconsistent and every later
        torch CUDA alloc (even the harness's 11 MB warmup dummy) trips an internal
        assert. The driver allocator has no such state. Best-effort; never fatal."""
        if "cuda" not in str(self.device):
            return
        # Make sure the device's primary CUDA context exists (TRT/pycuda/torch share it).
        try:
            import pycuda.autoprimaryctx  # noqa: F401
        except Exception:
            try:
                import pycuda.autoinit  # noqa: F401
            except Exception:
                pass
        import pycuda.driver as cuda
        for mb in (target_mb, target_mb * 3 // 4, target_mb // 2, target_mb // 3, 512):
            if mb <= 0:
                continue
            try:
                buf = cuda.mem_alloc(mb * 1024 * 1024)
                buf.free()
                print(f"NvMap defrag: reserved+freed {mb} MB before engine load.")
                return
            except Exception:
                continue
        print("NvMap defrag: could not reserve any block (proceeding).")

    @staticmethod
    def _sole_input_name(wrapper):
        eng, trt = wrapper.engine, wrapper._trt
        for i in range(eng.num_io_tensors):
            name = eng.get_tensor_name(i)
            if eng.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                return name
        raise RuntimeError("engine has no input binding")

    def forward(self, images: torch.Tensor, frames_chunk_size=None, **kwargs):
        # Stay entirely on the CPU/host in torch — the TRT engines own ALL of the GPU
        # (weights + shared scratch + their own pycuda I/O buffers). Any torch-CUDA
        # allocation here races the driver's last few hundred MB and trips the
        # CUDACachingAllocator NVML assert. infer() does its own H2D/D2H, so the
        # intermediate host arrays pass straight from one engine to the next.
        if images.dim() == 4:        # CO3D loader gives (S,C,H,W); dino_0 needs (B,S,C,H,W)
            images = images.unsqueeze(0)
        images = images.detach().cpu().float()   # .cpu() BEFORE .float(): no CUDA alloc
        if tuple(images.shape[-2:]) != self._in_hw:
            b, s = images.shape[:2]
            flat = images.reshape(b * s, *images.shape[2:])
            flat = torch.nn.functional.interpolate(
                flat, size=self._in_hw, mode="bilinear", align_corners=False)
            images = flat.reshape(b, s, *flat.shape[1:])
        # Pipe the running host array through every engine: images → … → pose_enc.
        x = images
        with torch.inference_mode():
            for eng, in_name, out_name in zip(self.engines, self._in_names, self._out_names):
                x = eng.infer({in_name: x})[out_name]   # numpy host array, fed straight on
        # Return on the eval device (cuda) like the other TRT models — the metric code
        # cats pose_enc with CUDA tensors. float() on CPU first (cheap), then move the
        # tiny (1,S,9) result.
        return {"pose_enc": torch.from_numpy(x).float().to(self.device)}

    def close(self):
        for e in self.engines:
            e.close()
        if getattr(self, "_shared_dm", None) is not None:
            try:
                self._shared_dm.free()
            except Exception:
                pass
            self._shared_dm = None


class TRTFullPoseModel(nn.Module):
    """Full-model pose engine: images [1,S,3,H,W] → pose_enc [1,S,9], ENTIRELY in
    TensorRT (DINO patch-embed + aggregator + camera head). No PyTorch compute — the
    whole pose path is the engine, so every non-quantized layer runs as TRT FP16 and
    only the QLAYERS are INT8 (for the int8 variants).

    Resizes eval frames to the fixed (H,W) the engine was exported for, exactly like
    TRTPoseModel, so TRT-vs-TRT comparisons stay apples-to-apples. No checkpoint is
    needed — there are no torch weights to load."""

    def __init__(self, engine_path: str, device: str = None,
                 dtype: torch.dtype = torch.float16, height: int = 350, width: int = 518,
                 **_ignored):
        super().__init__()
        self.engine = TRTInferenceEngine(engine_path)
        self._out_name = self.engine._output_names[0]   # "pose_enc"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._in_hw = (height, width)

    def forward(self, images: torch.Tensor, frames_chunk_size=None, **kwargs):
        images = images.to(self.device, self.dtype)
        if images.dim() == 4:        # CO3D loader gives (S,C,H,W); embed_tokens needs (B,S,C,H,W)
            images = images.unsqueeze(0)
        if tuple(images.shape[-2:]) != self._in_hw:
            b, s = images.shape[:2]
            flat = images.reshape(b * s, *images.shape[2:])
            flat = torch.nn.functional.interpolate(
                flat, size=self._in_hw, mode="bilinear", align_corners=False)
            images = flat.reshape(b, s, *flat.shape[1:])
        with torch.inference_mode():
            pose = self.engine.infer({"images": images})[self._out_name]
        pose_enc = torch.from_numpy(pose).to(self.device).float()
        return {"pose_enc": pose_enc}

    def close(self):
        self.engine.close()


# ──────────────────────────────────────────────────────────────────────────────
# Quick validation: compare TRT output vs PyTorch (FP32) baseline
# ──────────────────────────────────────────────────────────────────────────────

def validate_trt_vs_pytorch(
    engine_path: str,
    pytorch_model: torch.nn.Module,
    num_frames: int = 4,
    image_size: int = 224,
    tol: float = 0.05,
):
    """
    Runs the same random input through PyTorch (FP32) and TRT, prints max/mean error.
    tol is the acceptable max absolute error as a fraction of output value range.
    """
    dummy = torch.randn(1, num_frames, 3, image_size, image_size)

    pytorch_model.eval()
    with torch.no_grad():
        pt_out = pytorch_model(images=dummy)
    if isinstance(pt_out, dict):
        pt_arr = next(iter(pt_out.values())).numpy()
    else:
        pt_arr = pt_out.numpy()

    with TRTInferenceEngine(engine_path) as engine:
        trt_out = engine.infer({'images': dummy})
    trt_arr = next(iter(trt_out.values()))

    abs_err = np.abs(pt_arr - trt_arr)
    val_range = pt_arr.max() - pt_arr.min() + 1e-8
    rel_err = abs_err / val_range
    print(f'Max abs error: {abs_err.max():.4f}  '
          f'Mean abs error: {abs_err.mean():.4f}  '
          f'Max rel error: {rel_err.max()*100:.2f}%')
    if rel_err.max() > tol:
        print(f'WARNING: max relative error {rel_err.max()*100:.2f}% > tolerance {tol*100:.1f}%')
    else:
        print('Validation PASSED.')
