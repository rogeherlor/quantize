#!/usr/bin/env python3
"""Stage 1 — export the VGGT *pose* path (aggregator + camera_head → pose_enc)
to ONNX, and/or benchmark the PyTorch pose-only latency on Jetson.

Why pose-only: the CO3D AUC metric uses only predictions["pose_enc"] =
camera_head(aggregator(images))[-1]. The depth/point/track heads (heavy DPT
convs + frames_chunk_size chunking) are irrelevant to pose AUC, so we exclude
them — this is what makes the model tractable to export to ONNX/TensorRT.

This script does NOT touch the existing eval scripts. Run it inside the Jetson
container (dustynv/pytorch, /workspace).

Modes:
  --bench-torch          time the PyTorch fp16 pose-only forward (the fair
                         baseline — note eval_co3d.py's 12.5 s/seq includes the
                         depth head, which pose AUC does not use).
  --precision fp32|fp16  export the plain pose model to ONNX (one fp32 graph; the
                         engine precision is chosen later at build time).
  --precision w8a8       quantize the 12 QLAYERS with LSQU, calibrate the LSQ
                         scales on real CO3D frames, and export with explicit
                         Q/DQ nodes (TensorRT explicit quantization). Only the
                         same QLAYERS are quantized; everything else stays FP16.

Export runs on CPU in fp32: that auto-selects the aggregator's batched
patch-embed path (the GPU path processes one frame at a time → would unroll 10×
in the ONNX graph), avoids GPU OOM, and sidesteps fp16-CPU-op gaps. trtexec
applies fp16/int8 at build time.

Usage (in container):
  python scripts/trt_export_pose.py --bench-torch
  python scripts/trt_export_pose.py --precision fp16 --out deploy/engines/pose_fp16.onnx
"""
import os
import sys
import argparse
import contextlib
import logging
import subprocess
import time

# Match eval_co3d.py's Jetson allocator setup (harmless elsewhere).
os.environ.pop("PYTORCH_NO_CUDA_MEMORY_CACHING", None)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "")

import torch
import torch.nn as nn

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "../src/models/depth/vggt"))

from src.logger import logger
from vggt.models.vggt import VGGT

DEFAULT_CKPT = "data3/rogelio/model_zoo/vggt/vggt_1B_commercial.pt"
# Representative CO3D eval shape: load_and_preprocess_images → width 518, height
# ≈350 (divisible by 14). The aggregator bakes H,W into the traced graph, so the
# engine is built for this fixed shape (Stage 2 resizes eval images to match).
DEFAULT_FRAMES = 10
DEFAULT_H = 350
DEFAULT_W = 518


class VGGTPose(nn.Module):
    """Pose-only wrapper: images [B,S,3,H,W] → pose_enc [B,S,9]."""

    def __init__(self, vggt: VGGT):
        super().__init__()
        self.aggregator = vggt.aggregator
        self.camera_head = vggt.camera_head

    def forward(self, images):
        aggregated_tokens_list, _ = self.aggregator(images)
        return self.camera_head(aggregated_tokens_list)[-1]


class AggregatorBlocks(nn.Module):
    """Trace target for the blocks-only TensorRT engine.

    Wraps just the aggregator's alternating frame/global attention stack (the
    heavy, quantized part — the QLAYERS live here):

        tokens [B*S, P, C]  →  last-block tokens [B, S, P, 2C]

    DINO patch-embed + token assembly run upstream in PyTorch (Aggregator.embed_tokens)
    and the camera head runs downstream in PyTorch — both un-quantized and identical
    across every approach. The camera head only reads aggregated_tokens_list[-1], so a
    single output is sufficient. `pos` is baked as a constant buffer because the engine
    is fixed-shape (S, H, W all fixed)."""

    def __init__(self, aggregator, pos, B, S, P, C):
        super().__init__()
        self.aggregator = aggregator
        self.B, self.S, self.P, self.C = B, S, P, C
        if pos is not None:
            self.register_buffer("pos", pos)
        else:
            self.pos = None

    def forward(self, tokens):
        output_list, _ = self.aggregator.run_blocks(
            tokens, self.pos, self.B, self.S, self.P, self.C)
        return output_list[-1]


class AggregatorBlocksSlice(nn.Module):
    """One TensorRT SUB-engine = a contiguous range [start, end) of the aggregator's
    (frame, global) attention pairs.

    Splitting the 24-pair stack into N slices makes each sub-engine's fused weight
    constant ~1/N the size, so each builds within the Orin Nano's contiguous-memory
    limit (the monolithic blocks engine's ~1.15 GB weight block can't be allocated
    contiguously during the TRT build). At inference the slices are chained
    (TRTChainedPoseModel): the running token state (B*S, P, C) is the interface
    tensor between slices; the FINAL slice returns the camera-head input
    concat(last_frame_intermediate, last_global_intermediate) → [B, S, P, 2C]."""

    def __init__(self, aggregator, pos, B, S, P, C, start, end, is_last):
        super().__init__()
        self.aggregator = aggregator
        self.B, self.S, self.P, self.C = B, S, P, C
        self.start, self.end, self.is_last = start, end, is_last
        if pos is not None:
            self.register_buffer("pos", pos)
        else:
            self.pos = None

    def forward(self, tokens):
        agg = self.aggregator
        # Same global VGGT context the full run_blocks sets for the quant layers.
        import src.quantizer.uniform.lsqc as lsqc_module
        lsqc_module._vggt_num_frames = self.S
        lsqc_module._vggt_tokens_per_frame = self.P
        lsqc_module._vggt_batch_size = self.B
        frame_int = glob_int = None
        for idx in range(self.start, self.end):
            tokens, _, frame_int = agg._process_frame_attention(
                tokens, self.B, self.S, self.P, self.C, idx, pos=self.pos)
            tokens, _, glob_int = agg._process_global_attention(
                tokens, self.B, self.S, self.P, self.C, idx, pos=self.pos)
        if self.is_last:
            # camera head reads output_list[-1] = concat of the last pair → [B,S,P,2C]
            return torch.cat([frame_int[-1], glob_int[-1]], dim=-1)
        # hand the running token state to the next slice in a consistent (B*S,P,C) shape
        return tokens.reshape(self.B * self.S, self.P, self.C)


class DinoSlice(nn.Module):
    """One TRT sub-engine = a contiguous range [start,end) of the DINOv2 ViT-L's 24
    (flat) blocks. The whole ViT-L is a ~605 MB weight block that exceeds the Orin
    Nano's contiguous-build ceiling, so it's split like the aggregator. The interface
    between chunks is the ViT hidden state [B*S, T, dim].

    is_first: normalize images + reshape + prepare_tokens + blocks[start:end].
    is_last:  blocks[start:end] + final norm + token assembly (concat camera/register
              with the patch tokens) → tokens [B*S, P, C] (the transformer-chain input)."""

    def __init__(self, aggregator, B, S, start, end, is_first, is_last):
        super().__init__()
        self.agg = aggregator
        self.vit = aggregator.patch_embed   # the DinoVisionTransformer
        self.B, self.S = B, S
        self.start, self.end = start, end
        self.is_first, self.is_last = is_first, is_last

    def forward(self, x):
        agg, vit = self.agg, self.vit
        if self.is_first:
            # x = raw images [1, S, 3, H, W]
            x = (x - agg._resnet_mean) / agg._resnet_std
            x = x.reshape(self.B * self.S, *x.shape[2:])          # [B*S, 3, H, W]
            x = vit.prepare_tokens_with_masks(x)                  # [B*S, T, dim]
        for blk in vit.blocks[self.start:self.end]:
            x = blk(x)
        if self.is_last:
            x = vit.norm(x)
            patch_tokens = x[:, vit.num_register_tokens + 1:]     # strip ViT cls+register
            from vggt.models.aggregator import slice_expand_and_flatten
            cam = slice_expand_and_flatten(agg.camera_token, self.B, self.S)
            reg = slice_expand_and_flatten(agg.register_token, self.B, self.S)
            return torch.cat([cam, reg, patch_tokens], dim=1)     # [B*S, P, dim] = aggregator tokens
        return x


def _validate_onnx(onnx_path):
    """Validate the exported ONNX in a SUBPROCESS so the (multi-GB, external-data)
    load is reclaimed immediately and never stacks on top of the still-resident
    torch model — that stacking is what OOM-killed the Jetson before. check_model
    is called with the path (not a loaded proto) so >2 GB external-data graphs are
    handled without hitting the 2 GB protobuf limit. Best-effort: trtexec re-parses
    the graph at build time anyway, so a checker hiccup is non-fatal."""
    logger.info(f"Validating {onnx_path} (subprocess) ...")
    code = "import sys, onnx; onnx.checker.check_model(sys.argv[1]); print('ONNX_OK')"
    r = subprocess.run([sys.executable, "-c", code, onnx_path],
                       capture_output=True, text=True)
    if "ONNX_OK" in r.stdout:
        logger.info(f"ONNX validation passed: {onnx_path}")
    else:
        logger.warning("ONNX validation skipped/failed (non-fatal — trtexec re-parses): "
                       f"{(r.stderr or r.stdout).strip()[:500]}")


@contextlib.contextmanager
def _quiet_and_batched():
    """During export/trace: silence the aggregator's per-forward logging spam,
    and force the batched patch-embed path. The aggregator selects a per-frame
    loop when CUDA total_memory < 12 GB — that unrolls 10× in ONNX and also
    crashes for a CPU input (get_device_properties(cpu)). Faking a large device
    forces the single batched call → clean graph."""
    src_logger = logging.getLogger("src_logger")
    prev_level = src_logger.level
    src_logger.setLevel(logging.WARNING)

    class _FakeProps:
        total_memory = 64 * 1024 ** 3

    orig_props = torch.cuda.get_device_properties
    torch.cuda.get_device_properties = lambda *a, **k: _FakeProps()
    try:
        yield
    finally:
        torch.cuda.get_device_properties = orig_props
        src_logger.setLevel(prev_level)


def _build_pose_vggt():
    """VGGT with only the camera head enabled (no DPT depth/point, no track)."""
    return VGGT(enable_camera=True, enable_point=False, enable_depth=False, enable_track=False)


def _stream_state_dict(model, ckpt_path, device, dtype):
    """meta-build → mmap-stream matching keys → assign. Avoids the FP32+target
    double-peak and never materializes the full checkpoint in RAM at once."""
    sd = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    keys = set(k for k, _ in model.named_parameters()) | set(k for k, _ in model.named_buffers())
    streamed = {}
    for k in list(sd.keys()):
        t = sd[k]
        if isinstance(t, torch.Tensor) and k in keys:
            streamed[k] = t.to(device=device, dtype=dtype)
        del sd[k]
    missing, unexpected = model.load_state_dict(streamed, assign=True, strict=False)
    logger.info(f"Loaded pose model — missing: {len(missing)}, unexpected: {len(unexpected)} "
                f"(depth/point/track keys expected as unexpected)")
    # Non-persistent ImageNet norm buffers are not in the checkpoint.
    for m in model.modules():
        for name, val in (("_resnet_mean", [0.485, 0.456, 0.406]),
                          ("_resnet_std", [0.229, 0.224, 0.225])):
            buf = getattr(m, name, None)
            if buf is not None and (buf.is_meta or buf.device != torch.device(device) or buf.dtype != dtype):
                m.register_buffer(name, torch.tensor(val, dtype=dtype, device=device).view(1, 1, 3, 1, 1),
                                  persistent=False)
    return model


def _load_pose_cpu_fp32(ckpt_path):
    _orig_item = torch.Tensor.item
    torch.Tensor.item = lambda self: 0 if self.is_meta else _orig_item(self)
    try:
        with torch.device("meta"):
            model = _build_pose_vggt().eval()
    finally:
        torch.Tensor.item = _orig_item
    return _stream_state_dict(model, ckpt_path, device="cpu", dtype=torch.float32)


def export_baseline(args):
    """Export the plain (un-quantized) BLOCKS graph (tokens → last_tokens). One
    fp32 ONNX serves both the TRT fp32 and fp16 baselines — the engine precision is
    chosen later at build. DINO + camera head run in PyTorch at inference time."""
    import gc
    logger.info("Loading pose model on CPU (fp32) for export ...")
    model = _load_pose_cpu_fp32(args.checkpoint)
    agg = model.aggregator

    dummy = torch.zeros(1, args.frames, 3, args.height, args.width, dtype=torch.float32)
    logger.info("Running embed_tokens (DINO, CPU) once to produce the engine input ...")
    with _quiet_and_batched(), torch.no_grad():
        tokens, pos, B, S, P, C = agg.embed_tokens(dummy)
    logger.info(f"Engine input tokens={tuple(tokens.shape)}  "
                f"pos={None if pos is None else tuple(pos.shape)}")

    blocks = AggregatorBlocks(agg, pos, B, S, P, C).eval()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    logger.info(f"Exporting blocks ONNX (opset 17) tokens={tuple(tokens.shape)} → {args.out} ...")
    with _quiet_and_batched(), torch.no_grad():
        torch.onnx.export(
            blocks, (tokens,), args.out,
            opset_version=17,
            input_names=["tokens"], output_names=["last_tokens"],
            dynamic_axes=None,            # fixed shape → simpler, TRT-friendly graph
        )
    logger.info("ONNX export done.")
    # Free the torch model BEFORE validating (the prior OOM was onnx.load stacking
    # a second multi-GB copy on top of the still-resident model).
    del blocks, model, agg, tokens
    gc.collect()
    _validate_onnx(args.out)


def export_full(args):
    """Export the FULL pose path in ONE graph: images [1,S,3,H,W] → pose_enc
    [1,S,9] (DINO patch-embed + full aggregator + camera head). One fp32 ONNX; the
    engine precision (fp16) is chosen at build time.

    Unlike export_baseline (blocks-only → hybrid PyTorch+TRT), this puts the whole
    pose path in TensorRT, so the non-quantized parts run as TRT FP16 instead of
    eager PyTorch FP16. CPU/fp32 export keeps the batched patch-embed path and avoids
    GPU OOM (same rationale as the blocks export)."""
    import gc
    logger.info("Loading pose model on CPU (fp32) for FULL export ...")
    model = _load_pose_cpu_fp32(args.checkpoint)
    pose = VGGTPose(model).eval()

    dummy = torch.zeros(1, args.frames, 3, args.height, args.width, dtype=torch.float32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    logger.info(f"Exporting FULL pose ONNX (opset 17) images={tuple(dummy.shape)} → {args.out} ...")
    with _quiet_and_batched(), torch.no_grad():
        torch.onnx.export(
            pose, (dummy,), args.out,
            opset_version=17,
            input_names=["images"], output_names=["pose_enc"],
            dynamic_axes=None,            # fixed shape → simpler, TRT-friendly graph
        )
    logger.info("FULL ONNX export done.")
    del pose, model
    gc.collect()
    _validate_onnx(args.out)


def capture_lsq_scales(model, qlayers):
    """Snapshot the learned per-tensor LSQ weight/activation scales of every
    quantized QLinear in `qlayers` → {layer_name: {w_scale, x_scale}} (CPU tensors)."""
    import src.module_quantization as Q
    scales = {}
    for name, m in model.named_modules():
        if isinstance(m, Q.QLinear) and any(name == p or name.startswith(p + ".") for p in qlayers):
            w = m.w_Qparms.get("scale")
            x = m.x_Qparms.get("scale")
            scales[name] = {
                "w_scale": None if w is None else w.detach().cpu().clone(),
                "x_scale": None if x is None else x.detach().cpu().clone(),
            }
    return scales


def _inject_lsq_scales(model, scales):
    """Write captured scales into a freshly-structured (CPU) quantized model and
    freeze init_state so the first forward does not re-initialize from data."""
    import src.module_quantization as Q
    n = 0
    for name, m in model.named_modules():
        if name not in scales or not isinstance(m, Q.QLinear):
            continue
        s = scales[name]
        if s["w_scale"] is not None and m.w_Qparms.get("scale") is not None:
            tgt = m.w_Qparms["scale"]
            tgt.data = s["w_scale"].to(tgt.device, tgt.dtype)
        if s["x_scale"] is not None:
            tgt = m.x_Qparms.get("scale")
            if tgt is None:
                m.x_scale = torch.nn.Parameter(s["x_scale"].clone())
                m.x_Qparms["scale"] = m.x_scale
            else:
                tgt.data = s["x_scale"].to(tgt.device, tgt.dtype)
        m.init_state.fill_(True)
        n += 1
    logger.info(f"Injected LSQ scales into {n} CPU QLinear layers.")


# All 48 aggregator attention blocks (depth=24, frame + global). Quantising ALL of
# them — not just the 12 QLAYERS — shrinks the engine's fused weight constant enough
# that a SINGLE TRT engine may build within the Orin Nano's contiguous-memory limit.
ALL_BLOCKS = [f"aggregator.{t}_blocks.{i}" for t in ("frame", "global") for i in range(24)]


def _prepare_w8a8_export_model(args, qlayers=None, config="w8a8"):
    """Calibrate LSQ scales on GPU with the proven Jetson loader, then return a
    CPU-fp32 pose model whose `qlayers` QLinears are swapped to _ExportQLinear (emit
    QuantizeLinear/DequantizeLinear). `qlayers` defaults to the 12 QLAYERS; pass
    ALL_BLOCKS to quantise the whole aggregator. `config` is w8a8/w8a4/w4a4 (note: at
    opset 17 the Q/DQ container is int8 for all of these — only the range differs)."""
    import gc
    sys.path.insert(0, _HERE)
    import eval_co3d_realquant as erq
    from src.run_distill import selective_quantize_layers
    from deploy.export.onnx_exporter import _replace_qlinear_with_export

    if qlayers is None:
        qlayers = erq.QLAYERS
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. GPU: load full fp16 VGGT, drop non-pose heads (forward guards None),
    #    quantize the requested layers, and calibrate LSQ scales on real CO3D frames.
    logger.info(f"Loading full VGGT (fp16, Jetson pool) to calibrate {len(qlayers)} "
                f"blocks ({config}) ...")
    full = erq._load_model(args.checkpoint, torch.float16, device)
    full.depth_head = full.point_head = full.track_head = None
    gc.collect()
    full = selective_quantize_layers(full, erq._build_quant_args(config), qlayers)
    erq._defrag_for_inference(device, "post-quantize")
    erq._calibrate(full, torch.float16, device, num_frames=args.calib_frames,
                   num_seqs=args.calib_seqs, frames_chunk_size=2)
    scales = capture_lsq_scales(full, qlayers)
    logger.info(f"Captured LSQ scales for {len(scales)} QLinear layers.")
    del full
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # 2. CPU fp32: rebuild the pose model, re-apply quant structure, inject scales.
    logger.info("Rebuilding pose model on CPU (fp32) and injecting scales ...")
    model = _load_pose_cpu_fp32(args.checkpoint)
    model = selective_quantize_layers(model, erq._build_quant_args(config), qlayers)
    _inject_lsq_scales(model, scales)

    # 3. Swap QLinear → _ExportQLinear (emits Q/DQ).
    _replace_qlinear_with_export(model, qlayers)
    return model


def _count_qdq(onnx_path):
    """Count Q/DQ nodes from the graph proto only (header parse — does NOT load the
    multi-GB external weights)."""
    import onnx
    return sum(1 for nd in onnx.load(onnx_path, load_external_data=False).graph.node
               if nd.op_type in ("QuantizeLinear", "DequantizeLinear"))


def export_w8a8(args):
    """LSQ w8a8 → Q/DQ, BLOCKS-only graph (tokens → last_tokens) for the hybrid
    pipeline. DINO embed + camera head run in PyTorch around the engine."""
    import gc
    model = _prepare_w8a8_export_model(args)
    agg = model.aggregator

    dummy = torch.zeros(1, args.frames, 3, args.height, args.width, dtype=torch.float32)
    logger.info("Running embed_tokens (DINO, CPU) once to produce the engine input ...")
    with _quiet_and_batched(), torch.no_grad():
        tokens, pos, B, S, P, C = agg.embed_tokens(dummy)
    logger.info(f"Engine input tokens={tuple(tokens.shape)}  "
                f"pos={None if pos is None else tuple(pos.shape)}")

    blocks = AggregatorBlocks(agg, pos, B, S, P, C).eval()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    logger.info(f"Exporting QDQ blocks ONNX (opset 17) tokens={tuple(tokens.shape)} → {args.out} ...")
    with _quiet_and_batched(), torch.no_grad():
        torch.onnx.export(
            blocks, (tokens,), args.out,
            opset_version=17,
            input_names=["tokens"], output_names=["last_tokens"],
            dynamic_axes=None,
        )
    logger.info(f"QDQ export done: {_count_qdq(args.out)} Q/DQ nodes.")
    del blocks, model, agg, tokens
    gc.collect()
    _validate_onnx(args.out)


def export_full_w8a8(args):
    """LSQ w8a8 → Q/DQ on the FULL pose graph (images → pose_enc). The whole model
    is one TRT engine: every non-quantized layer (DINO embed, other aggregator
    blocks, camera head) runs as TRT FP16, and only the QLAYERS carry Q/DQ → INT8."""
    import gc
    model = _prepare_w8a8_export_model(args)
    pose = VGGTPose(model).eval()

    dummy = torch.zeros(1, args.frames, 3, args.height, args.width, dtype=torch.float32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    logger.info(f"Exporting FULL QDQ pose ONNX (opset 17) images={tuple(dummy.shape)} → {args.out} ...")
    with _quiet_and_batched(), torch.no_grad():
        torch.onnx.export(
            pose, (dummy,), args.out,
            opset_version=17,
            input_names=["images"], output_names=["pose_enc"],
            dynamic_axes=None,
        )
    logger.info(f"FULL QDQ export done: {_count_qdq(args.out)} Q/DQ nodes.")
    del pose, model
    gc.collect()
    _validate_onnx(args.out)


def export_quant_blocks_all(args, config):
    """BLOCKS Q/DQ graph quantising ALL 48 aggregator blocks at `config` (w8a8/w4a4),
    so the engine's fused weight constant shrinks (int8 container vs fp16) and a SINGLE
    engine may build within the Nano's contiguous-memory limit. Scales are LSQ-
    calibrated at export (PTQ-style → lower accuracy than the thesis's trained QAT);
    this is to obtain a buildable single engine to iterate from."""
    import gc
    model = _prepare_w8a8_export_model(args, qlayers=ALL_BLOCKS, config=config)
    agg = model.aggregator

    dummy = torch.zeros(1, args.frames, 3, args.height, args.width, dtype=torch.float32)
    logger.info("Running embed_tokens (DINO, CPU) once to produce the engine input ...")
    with _quiet_and_batched(), torch.no_grad():
        tokens, pos, B, S, P, C = agg.embed_tokens(dummy)

    blocks = AggregatorBlocks(agg, pos, B, S, P, C).eval()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    logger.info(f"Exporting ALL-blocks {config} QDQ ONNX (opset 17) "
                f"tokens={tuple(tokens.shape)} → {args.out} ...")
    with _quiet_and_batched(), torch.no_grad():
        torch.onnx.export(
            blocks, (tokens,), args.out,
            opset_version=17,
            input_names=["tokens"], output_names=["last_tokens"],
            dynamic_axes=None,
        )
    logger.info(f"ALL-blocks {config} export done: {_count_qdq(args.out)} Q/DQ nodes.")
    del blocks, model, agg, tokens
    gc.collect()
    _validate_onnx(args.out)


def export_slice(args):
    """Export ONE small FP16 sub-engine = aggregator pairs [slice_start, slice_start+
    slice_pairs). NO quantization → no calibration forward → no export OOM; tiny weight
    constant → builds on the Nano with room to spare. This is the minimal
    'it-builds-and-runs' example to start measuring per-block latency/memory from.

    is_last=True makes the output the camera-head shape [B,S,P,2C]; otherwise it returns
    the running token state [B*S,P,C] (for chaining)."""
    import gc
    logger.info("Loading pose model on CPU (fp32) for slice export ...")
    model = _load_pose_cpu_fp32(args.checkpoint)
    agg = model.aggregator

    dummy = torch.zeros(1, args.frames, 3, args.height, args.width, dtype=torch.float32)
    logger.info(f"Running embed_tokens (DINO, CPU, {args.frames} frame[s]) ...")
    with _quiet_and_batched(), torch.no_grad():
        tokens, pos, B, S, P, C = agg.embed_tokens(dummy)

    if getattr(args, "no_fused_attn", False):
        _disable_fused_attn(agg)
    start = args.slice_start
    end = min(start + args.slice_pairs, agg.aa_block_num)
    sl = AggregatorBlocksSlice(agg, pos, B, S, P, C, start, end, args.slice_is_last).eval()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_name = "last_tokens" if args.slice_is_last else "tokens_out"
    logger.info(f"Exporting slice pairs[{start}:{end}] is_last={args.slice_is_last} "
                f"in={tuple(tokens.shape)} → {args.out} ...")
    with _quiet_and_batched(), torch.no_grad():
        torch.onnx.export(
            sl, (tokens,), args.out,
            opset_version=17,
            input_names=["tokens"], output_names=[out_name],
            dynamic_axes=None,
        )
    logger.info("Slice export done.")
    del sl, model, agg, tokens
    gc.collect()
    _validate_onnx(args.out)


def export_dino(args):
    """Export the DINO patch-embed (DINOv2 ViT-L) + token assembly as its OWN engine:
    images [1,S,3,H,W] → tokens [B*S,P,C]. The ViT-L attention is per-frame (~P tokens),
    so it avoids the aggregator's 9300-token attention wall; the main cost is the
    ~600 MB ViT-L weight block (near the build ceiling — may need a split). Head of #3
    (full model all-TRT): THIS DINO engine → transformer-chain → camera-head engine."""
    import gc
    logger.info("Loading pose model on CPU (fp32) for DINO-embed export ...")
    model = _load_pose_cpu_fp32(args.checkpoint)
    agg = model.aggregator

    class _Dino(nn.Module):
        def __init__(self, a):
            super().__init__()
            self.a = a

        def forward(self, images):
            tokens, _pos, _B, _S, _P, _C = self.a.embed_tokens(images)
            return tokens   # [B*S, P, C] — pos is baked into the transformer engines

    dino = _Dino(agg).eval()
    dummy = torch.zeros(1, args.frames, 3, args.height, args.width, dtype=torch.float32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    logger.info(f"Exporting DINO-embed ONNX (opset 17) images={tuple(dummy.shape)} → {args.out} ...")
    with _quiet_and_batched(), torch.no_grad():
        torch.onnx.export(
            dino, (dummy,), args.out,
            opset_version=17,
            input_names=["images"], output_names=["tokens"],
            dynamic_axes=None,
        )
    logger.info("DINO-embed export done.")
    del dino, model, agg
    gc.collect()
    _validate_onnx(args.out)


def _disable_fused_attn(root):
    """Set fused_attn=False on every attention submodule so the export emits MANUAL
    attention (q@kᵀ → softmax → @v, plain MatMul/Softmax) instead of
    F.scaled_dot_product_attention. TRT's Myelin fusion of the SDPA pattern corrupts
    the result (confirmed on DINO via polygraphy: engine attenuated high-norm tokens
    140→74 vs ONNX-Runtime; the damage scales with activation magnitude). Returns count."""
    n = 0
    for m in root.modules():
        if hasattr(m, "fused_attn"):
            m.fused_attn = False
            n += 1
    logger.info(f"Disabled fused_attn (SDPA→manual) on {n} attention module(s) for export.")
    return n


def export_dino_split(args):
    """Export the DINO ViT-L as N sub-engine ONNX graphs (default 2) so each ~600 MB/N
    weight block fits the Nano. Chunk 0: images → ViT hidden state; last chunk: ViT
    hidden → tokens. Chained at inference (the head of the all-TRT pipeline, #3)."""
    import gc
    logger.info(f"Loading pose model on CPU (fp32) for {args.dino_chunks}-chunk DINO export ...")
    model = _load_pose_cpu_fp32(args.checkpoint)
    agg = model.aggregator
    if getattr(args, "no_fused_attn", False):
        _disable_fused_attn(agg.patch_embed)
    depth = len(agg.patch_embed.blocks)   # 24 ViT blocks
    B, S = 1, args.frames
    nck = args.dino_chunks
    bounds = [round(i * depth / nck) for i in range(nck + 1)]
    base, ext = os.path.splitext(args.out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    state = torch.zeros(1, args.frames, 3, args.height, args.width, dtype=torch.float32)
    for k in range(nck):
        start, end = bounds[k], bounds[k + 1]
        is_first, is_last = (k == 0), (k == nck - 1)
        sl = DinoSlice(agg, B, S, start, end, is_first, is_last).eval()
        out_k = f"{base}_{k}{ext}"
        in_name = "images" if is_first else "hidden_in"
        out_name = "tokens" if is_last else "hidden_out"
        logger.info(f"Exporting DINO chunk {k} blocks[{start}:{end}] is_first={is_first} "
                    f"is_last={is_last} in={tuple(state.shape)} → {out_k} ...")
        with _quiet_and_batched(), torch.no_grad():
            torch.onnx.export(
                sl, (state,), out_k,
                opset_version=17,
                input_names=[in_name], output_names=[out_name],
                dynamic_axes=None,
            )
        _validate_onnx(out_k)
        with _quiet_and_batched(), torch.no_grad():   # advance state for next chunk's trace
            state = sl(state)
    logger.info(f"Exported {nck} DINO chunk ONNX graphs ({base}_0..{nck-1}{ext}).")
    del model, agg
    gc.collect()


def export_camera_head(args):
    """Export the camera head as its OWN engine: last_tokens [B,S,P,2C] → pose_enc
    [B,S,9]. It reads only the camera token (tokens[:,:,0]) + a small 4-iteration
    trunk, so it builds trivially — it only failed when FUSED with the aggregator
    (the monolithic full-model engine). This is the tail of #3 (full model all-TRT
    via separate engines): DINO-engine → transformer-chain → THIS camera-head engine."""
    import gc
    logger.info("Loading pose model on CPU (fp32) for camera-head export ...")
    model = _load_pose_cpu_fp32(args.checkpoint)
    agg = model.aggregator

    # 1 frame just to read P (tokens/frame) and C (embed_dim) cheaply.
    probe = torch.zeros(1, 1, 3, args.height, args.width, dtype=torch.float32)
    with _quiet_and_batched(), torch.no_grad():
        _tok, _pos, _B, _S, P, C = agg.embed_tokens(probe)
    last_tokens = torch.zeros(1, args.frames, P, 2 * C, dtype=torch.float32)

    if getattr(args, "no_fused_attn", False):
        _disable_fused_attn(model.camera_head)

    class _CamHead(nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head

        def forward(self, last_tokens):
            return self.head([last_tokens])[-1]   # camera_head reads list[-1] → pose_enc

    cam = _CamHead(model.camera_head).eval()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    logger.info(f"Exporting camera-head ONNX (opset 17) last_tokens={tuple(last_tokens.shape)} "
                f"→ {args.out} ...")
    with torch.no_grad():
        torch.onnx.export(
            cam, (last_tokens,), args.out,
            opset_version=17,
            input_names=["last_tokens"], output_names=["pose_enc"],
            dynamic_axes=None,
        )
    logger.info("Camera-head export done.")
    del cam, model, agg
    gc.collect()
    _validate_onnx(args.out)


def bench_torch(args):
    """PyTorch fp16 pose-only latency on GPU — the fair baseline for the TRT
    comparison (excludes the depth head)."""
    import gc
    sys.path.insert(0, _HERE)
    import eval_co3d_realquant as erq   # reuse the proven Jetson fp16 loader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading full VGGT (fp16, Jetson pool strategy) for pose-only bench ...")
    full = erq._load_model(args.checkpoint, torch.float16, device)
    full.depth_head = None
    full.point_head = None
    full.track_head = None
    gc.collect()
    pose = VGGTPose(full).eval()

    images = torch.zeros(1, args.frames, 3, args.height, args.width, dtype=torch.float16, device=device)
    with torch.inference_mode():
        for _ in range(3):                      # warmup
            pose(images)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(args.bench_iters):
            t0 = time.perf_counter()
            pose(images)
            if device == "cuda":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    import numpy as np
    logger.info(f"PyTorch fp16 POSE-ONLY latency over {len(ts)} runs "
                f"(shape {tuple(images.shape)}):")
    logger.info(f"  mean={np.mean(ts)*1000:.1f} ms  median={np.median(ts)*1000:.1f} ms  "
                f"min={np.min(ts)*1000:.1f} ms  max={np.max(ts)*1000:.1f} ms")
    if device == "cuda":
        logger.info(f"  peak CUDA: {torch.cuda.max_memory_allocated()//1024**2} MB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--precision",
                   choices=["fp32", "fp16", "w8a8", "full", "full_w8a8",
                            "w8a8_all", "w4a4_all", "slice", "camera_head", "dino",
                            "dino_split"],
                   default="fp16",
                   help="fp32/fp16/w8a8 = BLOCKS graph, only the 12 QLAYERS quantized "
                        "(hybrid); full/full_w8a8 = FULL pose graph; w8a8_all/w4a4_all = "
                        "BLOCKS graph with ALL 48 aggregator blocks quantized; slice = ONE "
                        "small FP16 sub-engine (minimal, guaranteed to build — for first "
                        "measurements)")
    p.add_argument("--slice_start", type=int, default=0,
                   help="[slice] first aggregator (frame,global) pair to include")
    p.add_argument("--slice_pairs", type=int, default=1,
                   help="[slice] number of consecutive pairs in this sub-engine")
    p.add_argument("--slice_is_last", action="store_true",
                   help="[slice] emit camera-head output [B,S,P,2C] (else token state)")
    p.add_argument("--dino_chunks", type=int, default=2,
                   help="[dino_split] number of sub-engines to split the DINO ViT-L into")
    p.add_argument("--out", default="deploy/artifacts/onnx/pose_baseline/pose_baseline.onnx")
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    p.add_argument("--height", type=int, default=DEFAULT_H)
    p.add_argument("--width", type=int, default=DEFAULT_W)
    p.add_argument("--calib_seqs", type=int, default=2,
                   help="CO3D sequences for LSQ scale calibration (w8a8 export)")
    p.add_argument("--calib_frames", type=int, default=4,
                   help="frames per calibration sequence (w8a8 export)")
    p.add_argument("--bench-torch", action="store_true",
                   help="benchmark PyTorch fp16 pose-only latency instead of exporting")
    p.add_argument("--bench-iters", type=int, default=10)
    p.add_argument("--no_fused_attn", action="store_true",
                   help="[dino_split] export manual attention (q@kᵀ→softmax→@v) instead of "
                        "F.scaled_dot_product_attention, to dodge TRT's buggy SDPA Myelin fusion")
    args = p.parse_args()

    if args.bench_torch:
        bench_torch(args)
    elif args.precision in ("fp32", "fp16"):
        export_baseline(args)
    elif args.precision == "w8a8":
        export_w8a8(args)
    elif args.precision == "full":
        export_full(args)
    elif args.precision == "full_w8a8":
        export_full_w8a8(args)
    elif args.precision == "w8a8_all":
        export_quant_blocks_all(args, "w8a8")
    elif args.precision == "w4a4_all":
        export_quant_blocks_all(args, "w4a4")
    elif args.precision == "slice":
        export_slice(args)
    elif args.precision == "camera_head":
        export_camera_head(args)
    elif args.precision == "dino":
        export_dino(args)
    elif args.precision == "dino_split":
        export_dino_split(args)


if __name__ == "__main__":
    main()
