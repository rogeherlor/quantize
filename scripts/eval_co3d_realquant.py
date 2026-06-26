#!/usr/bin/env python3
"""Real-INT8 CO3D evaluation for quantized VGGT on Jetson (w8a8 / w8a4).

Mirrors scripts/eval_co3d.py (same Jetson memory strategy + same metric harness
run_evaluation_vggt) but executes the quantized model on real hardware:

  1. Load the full VGGT in FP16 (meta device + mmap streaming, like eval_co3d.py).
  2. Quantize the 12 aggregator blocks in QLAYERS in place (weights are SHARED by
     replace_all, so no extra GPU memory).
  3. Calibrate LSQ scales on a few REAL CO3D sequences (zeros would collapse the
     activation scales).
  4. Pack the quantized QLinear layers to real INT8 W8Linear (torch._int_mm).
  5. Run run_evaluation_vggt → same AUC / latency / memory JSON schema, so the
     result can be compared directly against the FP16 baseline.

Configs:
  w8a8  genuine 8-bit weights + 8-bit activations  (LSQU for both)
        → W8Linear; fully real INT8: real AUC + memory + speed vs FP16.
  w8a4  genuine 8-bit weights (real INT8) + 4-bit activations
        → W8Linear; weights int8, activations 4-bit-effective (LSQ, Qp=7).
  w4a4  4-bit weights + 4-bit activations
        → W4Linear: weights stored packed int4 (0.5 byte/weight, biggest memory
          win), executed via int8 tensor cores (unpacked W4→int8, int_mm).

Usage:
    python scripts/eval_co3d_realquant.py --config w8a8
    python scripts/eval_co3d_realquant.py --config w8a4 \
        --checkpoint data3/rogelio/model_zoo/vggt/vggt_1B_commercial.pt
"""
import os
import sys

# Disable expandable_segments: Jetson NvMap does not support VMM segment
# expansion (same rationale as scripts/eval_co3d.py).
os.environ.pop("PYTORCH_NO_CUDA_MEMORY_CACHING", None)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

# ── Drop page cache and compact physical memory before CUDA initialises ───────
try:
    with open("/proc/sys/vm/drop_caches", "w") as _f:
        _f.write("3\n")
except OSError:
    pass
try:
    with open("/proc/sys/vm/compact_memory", "w") as _f:
        _f.write("1\n")
    import time as _t; _t.sleep(3); del _t
except OSError:
    pass

import argparse
import gc
import json
import gzip
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/models/depth/vggt"))

from src.logger import logger
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from src.models.depth.qvggt import run_evaluation_vggt
from src.run_distill import selective_quantize_layers
from src.run_realquant import QLAYERS, pack_model_to_real_int, W8Linear, W4Linear
from src.quantizer.uniform import LSQU_quantizer, LSQ_quantizer
from src.initializer import LSQ_initializer

DEFAULT_CKPT = "data3/rogelio/model_zoo/vggt/vggt_1B_commercial.pt"
CO3D_DIR = "data3/rogelio/co3d/dataset/"
CO3D_ANNO_DIR = "data3/rogelio/co3d/preprocessed_dataset"
CALIB_CATEGORIES = ["apple", "backpack", "banana", "bench", "bicycle"]
# Activation memory scales ~130 MB/frame (Flash Attention, empirical).
ACT_POOL_MB = 1600


def _alloc_pool(target_mb, device, min_chunk_mb=32):
    chunks, pooled = [], 0
    chunk = 512
    while pooled < target_mb and chunk >= min_chunk_mb:
        want = min(chunk, target_mb - pooled)
        try:
            chunks.append(torch.empty(want * 1024 * 1024 // 4, dtype=torch.float32, device=device))
            pooled += want
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            chunk //= 2
    return chunks, pooled


def _load_model(checkpoint_path: str, dtype: torch.dtype, device: str) -> nn.Module:
    """Two-phase pool + meta device + mmap streaming — identical strategy to
    scripts/eval_co3d.py. Returns the FP16 VGGT with the activation pool already
    released back to the caching allocator's free list (contiguous, NvMap-free)."""
    try:
        with open("/proc/sys/vm/compact_memory", "w") as _f:
            _f.write("1\n")
        time.sleep(3)
    except OSError:
        pass

    cuda_free_mb = torch.cuda.mem_get_info()[0] // (1024 ** 2)
    model_pool_mb = min(2700, cuda_free_mb - ACT_POOL_MB - 200)

    logger.info(f"Pre-allocating model pool: {model_pool_mb} MB (kept alive) ...")
    _model_chunks, _model_pooled = _alloc_pool(model_pool_mb, device)

    logger.info(f"Pre-allocating activation pool: {ACT_POOL_MB} MB "
                f"(separate NvMap segments — model pool still alive) ...")
    _act_chunks, _act_pooled = _alloc_pool(ACT_POOL_MB, device)

    del _model_chunks
    gc.collect()
    logger.info(f"Pools ready: model {_model_pooled} MB + activation {_act_pooled} MB  "
                f"| CUDA free: {torch.cuda.mem_get_info()[0] // (1024**2)} MB")

    _orig_item = torch.Tensor.item
    torch.Tensor.item = lambda self: 0 if self.is_meta else _orig_item(self)
    try:
        with torch.device("meta"):
            model = VGGT().eval()
    finally:
        torch.Tensor.item = _orig_item

    logger.info(f"Streaming checkpoint ({dtype}) from {checkpoint_path} ...")
    state_dict = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    model_keys = (set(k for k, _ in model.named_parameters()) |
                  set(k for k, _ in model.named_buffers()))
    cuda_sd = {}
    for key in list(state_dict.keys()):
        t = state_dict[key]
        if isinstance(t, torch.Tensor) and key in model_keys:
            cuda_sd[key] = t.to(device, dtype=dtype)
        del state_dict[key]
    del state_dict
    gc.collect()

    missing, unexpected = model.load_state_dict(cuda_sd, assign=True, strict=False)
    del cuda_sd
    gc.collect()
    logger.info(f"Loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")

    try:
        with open("/proc/sys/vm/drop_caches", "w") as _f:
            _f.write("3\n")
        with open("/proc/sys/vm/compact_memory", "w") as _f:
            _f.write("1\n")
        time.sleep(2)
    except OSError:
        pass

    del _act_chunks
    gc.collect()
    logger.info(f"Activation pool released: {_act_pooled} MB contiguous free for inference  "
                f"| CUDA allocated: {torch.cuda.memory_allocated() // (1024**2)} MB")

    # Fix non-persistent ImageNet normalisation buffers not stored in checkpoint.
    for module in model.modules():
        for name, val in (("_resnet_mean", [0.485, 0.456, 0.406]),
                          ("_resnet_std",  [0.229, 0.224, 0.225])):
            buf = getattr(module, name, None)
            if buf is not None and (buf.is_meta or buf.device.type != device):
                module.register_buffer(
                    name,
                    torch.tensor(val, dtype=dtype, device=device).view(1, 1, 3, 1, 1),
                    persistent=False,
                )

    return model


def _build_quant_args(config: str) -> argparse.Namespace:
    """Quantizer config consumed by selective_quantize_layers.

    num_bits drives the packed container (pack_model_to_real_int: 8→W8Linear,
    4→W4Linear); the quantizer classes drive the effective ranges:
      w8a8: num_bits=8, w/x = LSQU      → W8Linear, weights 8b + activations 8b
      w8a4: num_bits=8, w=LSQU x=LSQ    → W8Linear, weights 8b + activations 4b
      w4a4: num_bits=4, w/x = LSQ       → W4Linear (packed int4), weights 4b + activations 4b
    All paths run real int8 tensor-core matmul via torch._int_mm (W4Linear
    unpacks int4→int8 first); w4a4 additionally halves weight storage.
    """
    args = argparse.Namespace()
    args.arch = "vggt"
    if config == "w8a8":
        args.num_bits, args.w_quantizer, args.x_quantizer = 8, LSQU_quantizer, LSQU_quantizer
    elif config == "w8a4":
        args.num_bits, args.w_quantizer, args.x_quantizer = 8, LSQU_quantizer, LSQ_quantizer
    elif config == "w4a4":
        args.num_bits, args.w_quantizer, args.x_quantizer = 4, LSQ_quantizer, LSQ_quantizer
    else:
        raise ValueError(f"unknown config: {config}")
    args.weight_norm = None
    args.w_grad_scale_mode = None
    args.x_grad_scale_mode = None
    args.w_initializer = LSQ_initializer
    args.x_initializer = LSQ_initializer
    # first/last layers are NOT in QLAYERS, so these are never instantiated;
    # present only because selective_quantize_layers builds the partials.
    args.first_bits = 8
    args.last_bits = 8
    args.w_first_last_quantizer = None
    args.x_first_last_quantizer = None
    args.w_first_last_initializer = None
    args.x_first_last_initializer = None
    args.w_first_last_grad_scale_mode = None
    args.x_first_last_grad_scale_mode = None
    return args


def _log_effective_bits(model: nn.Module):
    """Report the actual quantization range each QLAYER ended up with, so the
    effective weight/activation bit-widths are explicit (not just the label)."""
    import src.module_quantization as Q

    def _bits(qp):
        return int(round(np.log2((qp + 1) * 2))) if qp else None

    for name, m in model.named_modules():
        if isinstance(m, Q.QLinear) and any(name.startswith(p + ".") for p in QLAYERS):
            wb = _bits(int(m.w_Qp)) if hasattr(m, "w_Qp") else "?"
            xb = _bits(int(m.x_Qp)) if hasattr(m, "x_Qp") else "?"
            logger.info(f"Effective bits — first QLAYER '{name}': weights={wb}b "
                        f"(w_Qn={int(m.w_Qn)},w_Qp={int(m.w_Qp)}), "
                        f"activations={xb}b (x_Qn={int(m.x_Qn)},x_Qp={int(m.x_Qp)})")
            break


def _defrag_for_inference(device, label):
    """Re-establish a contiguous activation pool after mutating the model.

    selective_quantize_layers / pack_model_to_real_int allocate and free many
    transient GPU tensors (replace_all duplicates weights before sharing them;
    new scale params / int8 buffers). On Jetson's NvMap (no virtual memory) this
    fragments the contiguous free region that _load_model carefully built, so a
    later forward fails to find a contiguous block (NvMap ENOMEM / CUDA OOM).

    empty_cache returns the fragmented free blocks to NvMap, drop_caches/compact
    coalesce physical memory, and cycling an ACT_POOL_MB allocation pulls fresh
    contiguous segments back into the caching allocator's free list. Safe here
    (we are not yet inside the no-empty_cache inference loop)."""
    gc.collect()
    torch.cuda.empty_cache()
    try:
        with open("/proc/sys/vm/drop_caches", "w") as _f:
            _f.write("3\n")
        with open("/proc/sys/vm/compact_memory", "w") as _f:
            _f.write("1\n")
        time.sleep(2)
    except OSError:
        pass
    _chunks, _pooled = _alloc_pool(ACT_POOL_MB, device)
    del _chunks
    gc.collect()
    logger.info(f"Defrag [{label}]: cycled {_pooled} MB contiguous pool | "
                f"CUDA allocated: {torch.cuda.memory_allocated() // (1024**2)} MB  "
                f"free: {torch.cuda.mem_get_info()[0] // (1024**2)} MB")


def _cast_packed_to_dtype(model, dtype):
    """Align the float scale/bias of packed W8/W4 layers to the model dtype.

    pack_model_to_real_int creates fp32 scale buffers. On an fp16 model run
    without autocast (Jetson), an fp32 scale makes the layer emit fp32 outputs,
    which then break the next plain-fp16 layer (LayerNorm / unquantized blocks).
    Casting the float buffers — but NOT the int8/packed weights — keeps the whole
    forward in fp16 so quantized blocks are dtype-transparent to their neighbours.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, (W8Linear, W4Linear)):
            for name, buf in list(m._buffers.items()):
                if buf is not None and buf.is_floating_point() and buf.dtype != dtype:
                    m._buffers[name] = buf.to(dtype)
            if getattr(m, "bias", None) is not None and m.bias.is_floating_point():
                m.bias.data = m.bias.data.to(dtype)
            n += 1
    logger.info(f"Cast {n} packed INT layers' float scales/bias to {dtype}")


def _calibrate(model, dtype, device, num_frames=10, num_seqs=2, frames_chunk_size=2):
    """Run a few forward passes on REAL CO3D sequences so LSQ initializes
    representative weight/activation scales before packing. The init runs on the
    first forward of each QLinear and refines the scale (running max) on each
    subsequent pass (init_state is never frozen during fake-quant)."""
    logger.info(f"Calibrating LSQ scales on up to {num_seqs} real CO3D sequences "
                f"({num_frames} frames each) ...")
    done = 0
    for category in CALIB_CATEGORIES:
        if done >= num_seqs:
            break
        anno = os.path.join(CO3D_ANNO_DIR, f"{category}_test.jgz")
        try:
            with gzip.open(anno, "r") as fin:
                annotation = json.loads(fin.read())
        except FileNotFoundError:
            logger.info(f"  calib annotation not found: {os.path.abspath(anno)}")
            continue
        for seq_name in sorted(annotation.keys()):
            seq_data = annotation[seq_name]
            if len(seq_data) < num_frames:
                continue
            ids = np.random.choice(len(seq_data), num_frames, replace=False)
            image_names = [os.path.join(CO3D_DIR, seq_data[i]["filepath"]) for i in ids]
            try:
                images = load_and_preprocess_images(image_names).to(device=device, dtype=dtype)
            except Exception as e:
                logger.info(f"  skip {category}/{seq_name}: {e}")
                continue
            with torch.no_grad():
                model(images, frames_chunk_size=frames_chunk_size)
            del images
            gc.collect()  # drop activation refs (NOT empty_cache — keeps the NvMap pool)
            logger.info(f"  calibrated on {category}/{seq_name} ({num_frames} frames)")
            done += 1
            break
    if done == 0:
        logger.warning("No real calibration sequences found — activation scales "
                        "may be poorly initialized. Check CO3D dataset paths.")
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=["w8a8", "w8a4", "w4a4"], default="w8a8")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--results", default=None,
                   help="output JSON path (default: output/eval_results_<config>.json)")
    p.add_argument("--categories", default=None,
                   help="comma-separated category list for a fast estimate, e.g. "
                        "'apple' or 'apple,car'. Default: all 41 categories.")
    p.add_argument("--max_seqs", type=int, default=None,
                   help="cap sequences per category (e.g. 5) for a quick run.")
    p.add_argument("--pose_only", dest="pose_only", action="store_true", default=True,
                   help="drop depth/point/track heads (default) for head parity with the "
                        "TRT approaches and to avoid the point/track memory blow-up.")
    p.add_argument("--full_model", dest="pose_only", action="store_false",
                   help="keep all heads (original full-model behaviour).")
    p.add_argument("--calib_seqs", type=int, default=2)
    p.add_argument("--calib_frames", type=int, default=4,
                   help="frames per calibration sequence. Kept small: fake-quant adds "
                        "fp32 temporaries on top of full activations, and aggregator "
                        "activation memory scales with frame count, so 10 frames here "
                        "overflows the Jetson activation pool. Activation magnitudes are "
                        "frame-count-independent, so scales stay representative.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    frames_chunk_size = 2  # Jetson default (matches FP16 eval_co3d.py)
    results_path = args.results or f"output/eval_results_{args.config}.json"

    logger.info(f"Device: {device}  |  SM{torch.cuda.get_device_capability()[0]}.x  |  "
                f"dtype: {dtype}  |  config: {args.config}")

    # 1. Load full VGGT in FP16 (same Jetson strategy as the baseline).
    model = _load_model(args.checkpoint, dtype, device)

    # 1b. Pose-only (default): drop depth/point/track heads so the head config —
    #     and therefore weight memory — matches the TRT approaches exactly, and the
    #     point/track heads don't explode memory. VGGT.forward guards None heads.
    if args.pose_only:
        model.depth_head = model.point_head = model.track_head = None
        gc.collect()
        logger.info("Pose-only: depth/point/track heads disabled (head parity).")

    # 2. Quantize the 12 aggregator blocks in place (weights shared, no realloc).
    qargs = _build_quant_args(args.config)
    model = selective_quantize_layers(model, qargs, QLAYERS)
    _defrag_for_inference(device, "post-quantize")

    # 3. Calibrate LSQ scales on real data (zeros would collapse activation scales).
    _calibrate(model, dtype, device, num_frames=args.calib_frames,
               num_seqs=args.calib_seqs, frames_chunk_size=frames_chunk_size)
    _log_effective_bits(model)

    # 4. Pack to real INT8 (W8Linear, torch._int_mm). Dereferenced FP weights of
    #    the packed blocks return to the allocator free list via gc (NOT
    #    empty_cache, which would destroy the NvMap activation pool on Jetson).
    model = pack_model_to_real_int(model, QLAYERS)
    _cast_packed_to_dtype(model, dtype)
    logger.info(f"Packed to real INT8 | CUDA allocated: "
                f"{torch.cuda.memory_allocated() // (1024**2)} MB")
    _defrag_for_inference(device, "post-pack")

    # 5. Same metric harness as the FP16 baseline → comparable JSON.
    categories = [c.strip() for c in args.categories.split(",")] if args.categories else None
    run_evaluation_vggt(
        model,
        model_path=f"{args.checkpoint} [{args.config}]",
        frames_chunk_size=frames_chunk_size,
        results_path=results_path,
        categories=categories,
        max_seqs=args.max_seqs,
    )


if __name__ == "__main__":
    main()
