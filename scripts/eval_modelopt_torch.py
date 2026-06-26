#!/usr/bin/env python3
"""Comparison point #2 — ModelOpt INT8 PTQ evaluated in PyTorch (fake-quant), no TensorRT.

Quantizes ONLY the 12 transformer QLAYERS with NVIDIA ModelOpt, calibrates on real CO3D
frames, then runs the SHARED CO3D pose-AUC harness (run_evaluation_vggt) on the fake-quant
model. This measures the INT8 *quantization quality* itself, independent of any runtime
(the Q/DQ math is simulated in fp16/fp32). The SAME calibration config feeds the TensorRT
INT8 engine (scripts/bench_trt_modelopt.py), so comparing this number to the TRT one shows
whether TensorRT preserves the quantization.

Pipeline is the hybrid transformer-only setup: DINO embed + camera head stay FP16 torch;
only the aggregator's QLAYERS are INT8.

Usage (inside the Jetson container):
  python scripts/eval_modelopt_torch.py --algorithm max --categories apple --max_seqs 2
  python scripts/eval_modelopt_torch.py --algorithm smoothquant --weight_per_tensor
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

import trt_export_pose as tep
import eval_co3d_realquant as erq
import export_modelopt_ptq as mo
from src.logger import logger
from src.models.depth.qvggt import run_evaluation_vggt

# ModelOpt fake-quant inserts quantize/dequantize temporaries on the QLAYERS, so the
# allocator must GROW past the pre-allocated activation pool. With the torch-2.7 default
# (expandable_segments), that growth queries NVML — unsupported on Tegra → the
# `NVML_SUCCESS == r` assert. Force expandable_segments:False (uses cudaMemGetInfo, which
# works on Tegra; JETSON_PLAN §10.8). MUST come AFTER `import eval_co3d_realquant`, which
# resets this var to "" at its import, and BEFORE the first CUDA op (in main()).
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=tep.DEFAULT_CKPT)
    p.add_argument("--algorithm", choices=mo.CALIB_METHODS, default="max")
    p.add_argument("--weight_per_channel", dest="weight_per_channel", action="store_true",
                   default=True, help="per-output-channel weight scales (default)")
    p.add_argument("--weight_per_tensor", dest="weight_per_channel", action="store_false",
                   help="per-tensor weight scales")
    p.add_argument("--calib_seqs", type=int, default=2)
    p.add_argument("--calib_frames", type=int, default=4)
    p.add_argument("--categories", default=None)
    p.add_argument("--max_seqs", type=int, default=None)
    p.add_argument("--results", default=None)
    p.add_argument("--act_pool_mb", type=int, default=2600,
                   help="contiguous activation pool MB (fake-quant needs more than fp16's 1600)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"{args.algorithm}_{'pc' if args.weight_per_channel else 'pt'}"

    # ModelOpt fake-quant adds quantize/dequantize temporaries on the QLAYERS, so inference
    # needs a larger contiguous activation pool than plain fp16 (the default 1600 MB OOMs →
    # NvMap error 12). The pool is freed before inference, so this just reserves more
    # contiguous headroom; bump it (still leaves room for the 2700 MB model pool at setup).
    erq.ACT_POOL_MB = args.act_pool_mb

    # Pose-only full model (depth/point/track heads off) — same hybrid setup as the TRT path.
    logger.info(f"Loading full VGGT (fp16) for ModelOpt fake-quant eval [{tag}] "
                f"(act pool {erq.ACT_POOL_MB} MB) ...")
    full = erq._load_model(args.checkpoint, torch.float16, device)
    full.depth_head = full.point_head = full.track_head = None
    gc.collect()

    # Insert INT8 quantizers on the QLAYERS + calibrate (shared helper → matches TRT export).
    mo.quantize_model(full, calib_seqs=args.calib_seqs, calib_frames=args.calib_frames,
                      algorithm=args.algorithm, weight_per_channel=args.weight_per_channel,
                      device=device)
    gc.collect()   # drop calibration temporaries before the eval forward

    cats = [c.strip() for c in args.categories.split(",")] if args.categories else None
    results = args.results or os.path.join(
        "deploy/artifacts/results", f"eval_results_modelopt_torch_{tag}.json")
    os.makedirs(os.path.dirname(results) or ".", exist_ok=True)
    logger.info(f"Evaluating ModelOpt fake-quant INT8 [{tag}] → {results}")
    run_evaluation_vggt(full, model_path=f"ModelOpt INT8 fake-quant ({tag})",
                        results_path=results, categories=cats, max_seqs=args.max_seqs)


if __name__ == "__main__":
    main()
