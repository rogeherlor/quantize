#!/usr/bin/env python3
"""C2 — NVIDIA TensorRT Model Optimizer (nvidia-modelopt) INT8 PTQ on the VGGT
aggregator BLOCKS graph (tokens → last_tokens), restricted to the SAME 12 QLAYERS
as the custom LSQ approach, exported as standardized Q/DQ ONNX.

This is the standardized-tool comparison point for the custom LSQ Q/DQ export
(`trt_export_pose.py --precision w8a8`). Both produce a blocks ONNX where only the
QLAYERS carry QuantizeLinear/DequantizeLinear → INT8 and everything else stays FP16
in the engine; they differ ONLY in where the INT8 scales come from (ModelOpt
max-calibration here vs learned LSQ there).

Why blocks and not the full pose graph: the full `images → pose_enc` engine does not
build on the Orin Nano (7.4 GB) — the DINO-embed weights starve TRT of build-time
scratch and the camera-head fusion can't be placed. The blocks engine is the proven,
buildable path (DINO embed + camera head run in PyTorch FP16 around it), and the
ENTIRE transformer (aggregator) still runs in TRT. ModelOpt is calibrated on the full
model (real CO3D frames flow through DINO→aggregator) so the QLAYERS see real token
statistics, then only the blocks subgraph is exported.

Run on the Jetson (needs `nvidia-modelopt` + the CO3D dataset). Install WITHOUT the
[torch] extra so it can't clobber the JetPack torch:
  pip install nvidia-modelopt --no-deps
  python scripts/export_modelopt_ptq.py \
      --out deploy/artifacts/onnx/pose_modelopt/pose_modelopt.onnx

Then build the engine with deploy/export/trt_builder.py (precision int8, no
calibrator — scales are baked into the Q/DQ nodes), exactly like the LSQ ONNX.

NOTE: ModelOpt's Python API has drifted across versions; the quant-config filter
keys (`*weight_quantizer` / `*input_quantizer`) and `mtq.quantize` signature here
follow the 0.1x+ convention. Adjust if your installed version differs.
"""
import argparse
import gc
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "../src/models/depth/vggt"))

from src.logger import logger
import trt_export_pose as tep   # VGGTPose, _quiet_and_batched, _count_qdq, _validate_onnx
import eval_co3d_realquant as erq   # QLAYERS, _load_model, CO3D loaders

# ModelOpt fake-quant calibration grows the allocator past the pre-allocated pool; on Tegra
# the default expandable_segments path then queries NVML (unsupported) → assert. Force
# expandable_segments:False (cudaMemGetInfo). MUST be after `import eval_co3d_realquant`
# (resets this var to "") and before the first CUDA op.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"


def _calib_image_batches(num_seqs, num_frames, device, dtype):
    """Yield up to `num_seqs` real CO3D image tensors [1,num_frames,3,H,W] — the
    same data source as the LSQ calibration (_calibrate), so the two INT8 variants
    see identical statistics."""
    import gzip, json
    import numpy as np
    from vggt.utils.load_fn import load_and_preprocess_images
    np.random.seed(0)   # deterministic calib frames → #2 (fake-quant) and #3 (TRT) match
    done = 0
    for category in erq.CALIB_CATEGORIES:
        if done >= num_seqs:
            break
        anno = os.path.join(erq.CO3D_ANNO_DIR, f"{category}_test.jgz")
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
            image_names = [os.path.join(erq.CO3D_DIR, seq_data[i]["filepath"]) for i in ids]
            try:
                images = load_and_preprocess_images(image_names).to(device=device, dtype=dtype)
            except Exception as e:
                logger.info(f"  skip {category}/{seq_name}: {e}")
                continue
            yield images.unsqueeze(0) if images.dim() == 4 else images
            done += 1
            break
    if done == 0:
        logger.warning("No CO3D calibration sequences found — check dataset paths.")


# Calibration methods exposed by the installed ModelOpt (0.44) that suit W8A8 PTQ.
# (AWQ/GPTQ are weight-only-oriented; entropy/percentile from the old pytorch-quantization
# toolkit are gone.) smoothquant migrates activation outliers into the weights.
CALIB_METHODS = ("max", "mse", "smoothquant")


def build_quant_cfg(algorithm="max", weight_per_channel=True):
    """INT8 W8A8 config in ModelOpt 0.44's LIST schema: disable every quantizer, then enable
    + configure ONLY the 12 QLAYERS' weight+input quantizers (same layer set as the LSQ path).
    Later list entries override earlier ones, so the leading '*' disable is overridden for the
    QLAYERS. `algorithm` is the calibration method; weights are per-output-channel (axis 0) or
    per-tensor (None); activations stay per-tensor.

    NOTE: schema changed across ModelOpt versions — 0.44 wants a list of
    {quantizer_name, cfg|enable}; older releases used a {glob: cfg} dict. Adjust if needed."""
    waxis = 0 if weight_per_channel else None
    quant_cfg = [{"quantizer_name": "*", "enable": False}]
    for q in erq.QLAYERS:
        quant_cfg.append({"quantizer_name": f"*{q}*weight_quantizer",
                          "cfg": {"num_bits": 8, "axis": waxis}})
        quant_cfg.append({"quantizer_name": f"*{q}*input_quantizer",
                          "cfg": {"num_bits": 8, "axis": None}})
    algo = "max" if algorithm == "max" else {"method": algorithm}
    return {"quant_cfg": quant_cfg, "algorithm": algo}


def quantize_model(model, *, calib_seqs, calib_frames, algorithm="max",
                   weight_per_channel=True, device="cuda", dtype=torch.float16):
    """Insert ModelOpt INT8 quantizers on the 12 QLAYERS and calibrate on real CO3D frames.
    `model` is BOTH quantized in place and driven through the calibration forward loop (the
    full pose model for eval, or VGGTPose for export — same QLAYERS either way). Reused by
    export_modelopt() and scripts/eval_modelopt_torch.py so #2 (fake-quant) and #3 (TRT) share
    identical quantization."""
    import modelopt.torch.quantization as mtq
    cfg = build_quant_cfg(algorithm, weight_per_channel)

    def forward_loop(m):
        with torch.no_grad(), tep._quiet_and_batched():
            for images in _calib_image_batches(calib_seqs, calib_frames, device, dtype):
                m(images)
                del images
                gc.collect()

    logger.info(f"ModelOpt INT8 PTQ: algorithm={algorithm}, weights="
                f"{'per-channel' if weight_per_channel else 'per-tensor'}, "
                f"on {len(erq.QLAYERS)} QLAYERS ...")
    mtq.quantize(model, cfg, forward_loop=forward_loop)
    if hasattr(mtq, "print_quant_summary"):
        try:
            mtq.print_quant_summary(model)
        except Exception:
            pass
    return model


def _require_modelopt():
    try:
        import modelopt.torch.quantization  # noqa: F401
    except ImportError:
        raise ImportError(
            "nvidia-modelopt is not installed. Install WITHOUT deps so it can't clobber the\n"
            "JetPack torch (see requirements.jetson.txt):\n"
            "  pip install nvidia-modelopt --no-deps --index-url https://pypi.org/simple/")


def _load_and_quantize(args, device):
    """Load the pose-only VGGT, insert ModelOpt INT8 quantizers on the QLAYERS, calibrate.
    Shared by the monolithic export, the sliced (chain) export, and the fake-quant eval, so
    every INT8 artifact uses identical quantization.

    device='cpu' is the CRASH-SAFE path: the calibration forward runs on CPU/fp32 (cushioned
    by swap), so it physically cannot GPU-OOM the Tegra (the full-model fake-quant forward on
    the 7.6 GB GPU hard-OOMs and takes down the container). Slower, but the INT8 scales from
    CPU-fp32 activation stats are valid for the GPU TRT engine."""
    if device == "cpu":
        logger.info("Loading pose model on CPU/fp32 for ModelOpt PTQ (GPU-free, crash-safe) ...")
        full = tep._load_pose_cpu_fp32(args.checkpoint)
        dtype = torch.float32
    else:
        # GPU path: the fake-quant calib forward needs more contiguous activation headroom than
        # plain fp16 (default 1600 MB OOMs → NvMap error 12). The pool is freed after load.
        erq.ACT_POOL_MB = getattr(args, "act_pool_mb", 2200)
        logger.info(f"Loading full VGGT (fp16, Jetson pool, act {erq.ACT_POOL_MB} MB) for ModelOpt PTQ ...")
        full = erq._load_model(args.checkpoint, torch.float16, device)
        dtype = torch.float16
    full.depth_head = full.point_head = full.track_head = None
    gc.collect()
    pose = tep.VGGTPose(full).eval()
    quantize_model(pose, calib_seqs=args.calib_seqs, calib_frames=args.calib_frames,
                   algorithm=args.algorithm, weight_per_channel=args.weight_per_channel,
                   device=device, dtype=dtype)
    return pose, full


def export_modelopt(args):
    _require_modelopt()
    device = "cpu" if getattr(args, "cpu", False) else ("cuda" if torch.cuda.is_available() else "cpu")
    pose, full = _load_and_quantize(args, device)

    # 3. Move to CPU/fp32 for a clean, OOM-safe export (same rationale as the LSQ
    #    path: CPU/fp32 keeps the batched patch-embed path and avoids GPU OOM). The
    #    GPU copy is freed first so both copies don't co-reside.
    logger.info("Moving quantized model to CPU (fp32) for export ...")
    pose.to("cpu", torch.float32)
    if device == "cuda":
        torch.cuda.empty_cache()

    # 4. Export the BLOCKS subgraph (tokens → last_tokens) — the quantized QLAYERS
    #    live in the aggregator, so their ModelOpt Q/DQ nodes are captured here.
    agg = pose.aggregator
    dummy = torch.zeros(1, args.frames, 3, args.height, args.width, dtype=torch.float32)
    logger.info("Running embed_tokens (DINO, CPU) once to produce the engine input ...")
    with tep._quiet_and_batched(), torch.no_grad():
        tokens, pos, B, S, P, C = agg.embed_tokens(dummy)
    # Free everything the blocks export does NOT use (DINO patch-embed ~1.2 GB fp32 + camera
    # head). torch.onnx.export of the fp32 model transiently ~doubles RAM, so on the 7.6 GB
    # unified board the full model OOM-kills (exit 137); the blocks graph only needs run_blocks.
    agg.patch_embed = None
    if getattr(full, "camera_head", None) is not None:
        full.camera_head = None
    del dummy
    gc.collect()
    blocks = tep.AggregatorBlocks(agg, pos, B, S, P, C).eval()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    logger.info(f"Exporting ModelOpt Q/DQ blocks ONNX (opset 17) tokens={tuple(tokens.shape)} → {args.out} ...")
    with tep._quiet_and_batched(), torch.no_grad():
        torch.onnx.export(
            blocks, (tokens,), args.out,
            opset_version=17,
            input_names=["tokens"], output_names=["last_tokens"],
            dynamic_axes=None,
        )
    logger.info(f"ModelOpt export done: {tep._count_qdq(args.out)} Q/DQ nodes.")
    del blocks, pose, full, agg, tokens
    gc.collect()
    tep._validate_onnx(args.out)


def export_modelopt_chain(args):
    """Export the quantized aggregator as N SLICES (8 pairs each) instead of one monolithic
    blocks graph. The monolithic fp32 onnx.export OOM-kills the 7.6 GB board (proto ~doubles
    the 3.2 GB aggregator); each slice's proto is only ~1/N, so peak stays ~6 GB. Mirrors the
    fp16 chain (trt_chain_build.sh / AggregatorBlocksSlice). Writes <base>_s{start}{ext} per
    slice; returns the list. Calibrate ONCE on the full model, then slice."""
    _require_modelopt()
    device = "cpu" if getattr(args, "cpu", False) else ("cuda" if torch.cuda.is_available() else "cpu")
    pose, full = _load_and_quantize(args, device)

    logger.info("Moving quantized model to CPU (fp32) for sliced export ...")
    pose.to("cpu", torch.float32)
    if device == "cuda":
        torch.cuda.empty_cache()
    agg = pose.aggregator
    dummy = torch.zeros(1, args.frames, 3, args.height, args.width, dtype=torch.float32)
    logger.info("Running embed_tokens (DINO, CPU) once for the first slice's input ...")
    with tep._quiet_and_batched(), torch.no_grad():
        tokens, pos, B, S, P, C = agg.embed_tokens(dummy)
    agg.patch_embed = None
    if getattr(full, "camera_head", None) is not None:
        full.camera_head = None
    del dummy
    gc.collect()

    total = agg.aa_block_num
    chunk = args.chunk_pairs
    base, ext = os.path.splitext(args.out)
    outs = []
    state = tokens
    with tep._quiet_and_batched(), torch.no_grad():
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            is_last = (end == total)
            sl = tep.AggregatorBlocksSlice(agg, pos, B, S, P, C, start, end, is_last).eval()
            out_k = f"{base}_s{start}{ext}"
            os.makedirs(os.path.dirname(out_k) or ".", exist_ok=True)
            out_name = "last_tokens" if is_last else "tokens_out"
            logger.info(f"Exporting quantized slice pairs[{start}:{end}] is_last={is_last} "
                        f"in={tuple(state.shape)} → {out_k} ...")
            torch.onnx.export(sl, (state,), out_k, opset_version=17,
                              input_names=["tokens"], output_names=[out_name], dynamic_axes=None)
            tep._validate_onnx(out_k)
            outs.append(out_k)
            state = sl(state)   # advance the running token state for the next slice's trace
    logger.info(f"ModelOpt chain export done: {len(outs)} slices "
                f"({tep._count_qdq(outs[0])} Q/DQ in slice 0).")
    del pose, full, agg, tokens, state
    gc.collect()
    return outs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=tep.DEFAULT_CKPT)
    p.add_argument("--out",
                   default="deploy/artifacts/onnx/pose_modelopt/pose_modelopt.onnx")
    p.add_argument("--frames", type=int, default=tep.DEFAULT_FRAMES)
    p.add_argument("--height", type=int, default=tep.DEFAULT_H)
    p.add_argument("--width", type=int, default=tep.DEFAULT_W)
    p.add_argument("--calib_seqs", type=int, default=2)
    p.add_argument("--calib_frames", type=int, default=4)
    p.add_argument("--algorithm", choices=CALIB_METHODS, default="max",
                   help="ModelOpt calibration method for the INT8 scales")
    p.add_argument("--weight_per_channel", dest="weight_per_channel", action="store_true",
                   default=True, help="per-output-channel weight scales (default)")
    p.add_argument("--weight_per_tensor", dest="weight_per_channel", action="store_false",
                   help="per-tensor weight scales")
    p.add_argument("--chain", action="store_true",
                   help="export N sliced ONNX (fits the 7.6 GB board) instead of one blocks graph")
    p.add_argument("--chunk_pairs", type=int, default=8,
                   help="[--chain] aggregator pairs per slice (24 total → 3 slices at 8)")
    p.add_argument("--act_pool_mb", type=int, default=2200,
                   help="contiguous activation pool MB for the calib forward (fake-quant needs >1600)")
    p.add_argument("--cpu", action="store_true",
                   help="run load+quantize+calibrate on CPU/fp32 (crash-safe: cannot GPU-OOM the "
                        "Tegra; slower, cushioned by swap). Recommended on the 7.6 GB Orin.")
    args = p.parse_args()
    if args.chain:
        export_modelopt_chain(args)
    else:
        export_modelopt(args)


if __name__ == "__main__":
    main()
