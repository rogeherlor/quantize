#!/usr/bin/env python3
"""Approach 6 — NVIDIA ModelOpt PTQ → QDQ → TensorRT INT8 (blocks engine).

The standardized-tool counterpart to bench_trt_lsq.py: same blocks graph, same 12
QLAYERS quantized, same hybrid pipeline (PyTorch DINO embed + TRT blocks + PyTorch
camera head). The ONLY difference vs LSQ is the INT8 scale source — NVIDIA
TensorRT Model Optimizer max-calibration instead of learned LSQ scales.

Needs `nvidia-modelopt` (install WITHOUT the [torch] extra on Jetson):
  pip install nvidia-modelopt --no-deps

Usage (inside the Jetson container):
  python scripts/bench_trt_modelopt.py --categories apple --max_seqs 2
  python scripts/bench_trt_modelopt.py
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

import _trt_bench_common as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=C.DEFAULT_CKPT)
    p.add_argument("--onnx", default=None)
    p.add_argument("--engine", default=None)
    p.add_argument("--results", default=None)
    p.add_argument("--categories", default=None)
    p.add_argument("--max_seqs", type=int, default=None)
    p.add_argument("--calib_seqs", type=int, default=2)
    p.add_argument("--calib_frames", type=int, default=4)
    p.add_argument("--algorithm", choices=("max", "mse", "smoothquant"), default="max",
                   help="ModelOpt calibration method (must match the fake-quant eval)")
    p.add_argument("--weight_per_channel", dest="weight_per_channel", action="store_true",
                   default=True)
    p.add_argument("--weight_per_tensor", dest="weight_per_channel", action="store_false")
    p.add_argument("--workspace_gb", type=float, default=4.0)
    args = p.parse_args()

    # Tag artifacts by calib config so different sweeps don't collide.
    tag = f"pose_modelopt_{args.algorithm}_{'pc' if args.weight_per_channel else 'pt'}"
    onnx = args.onnx or C.onnx_path(tag)
    engine = args.engine or C.engine_path(tag)
    results = args.results or C.results_path("trt_" + tag.replace("pose_", ""))

    # 1. Export the ModelOpt QDQ blocks ONNX (calibrates on real CO3D frames).
    C.ensure_modelopt_onnx(onnx, checkpoint=args.checkpoint,
                           calib_seqs=args.calib_seqs, calib_frames=args.calib_frames,
                           algorithm=args.algorithm, weight_per_channel=args.weight_per_channel)

    # 2. Build INT8 engine — scales come from the Q/DQ nodes, no calibrator.
    from deploy.export.trt_builder import build_engine
    build_engine(onnx, engine, precision="int8",
                 workspace_gb=args.workspace_gb, timing_cache=C.timing_cache_path())

    # 3. Evaluate → comparable JSON (hybrid blocks model).
    C.eval_engine(engine, results,
                  label=f"TRT INT8 ModelOpt ({args.algorithm},"
                        f"{'pc' if args.weight_per_channel else 'pt'})",
                  categories=args.categories, max_seqs=args.max_seqs,
                  checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
