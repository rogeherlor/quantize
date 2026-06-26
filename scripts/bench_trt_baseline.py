#!/usr/bin/env python3
"""Approaches 2 & 3 — TensorRT FP32 / FP16 pose baselines.

Builds a TRT engine from the plain (un-quantized) pose ONNX and evaluates it with
the same CO3D pose-AUC harness as every other approach, writing a comparable JSON.

Usage (inside the Jetson container):
  python scripts/bench_trt_baseline.py --precision fp16
  python scripts/bench_trt_baseline.py --precision fp32 --categories apple --max_seqs 2
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
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    p.add_argument("--checkpoint", default=C.DEFAULT_CKPT)
    p.add_argument("--onnx", default=None)
    p.add_argument("--engine", default=None)
    p.add_argument("--results", default=None)
    p.add_argument("--categories", default=None)
    p.add_argument("--max_seqs", type=int, default=None)
    p.add_argument("--workspace_gb", type=float, default=4.0)
    args = p.parse_args()

    onnx = args.onnx or C.onnx_path("pose_baseline")
    engine = args.engine or C.engine_path(f"pose_{args.precision}")
    results = args.results or C.results_path(f"trt_{args.precision}")

    # 1. One fp32 ONNX serves both fp32 and fp16 engines.
    C.ensure_baseline_onnx(onnx, checkpoint=args.checkpoint)

    # 2. Build the engine at the requested precision.
    from deploy.export.trt_builder import build_engine
    build_engine(onnx, engine, precision=args.precision,
                 workspace_gb=args.workspace_gb, timing_cache=C.timing_cache_path())

    # 3. Evaluate (latency + memory + pose AUC) → comparable JSON.
    C.eval_engine(engine, results, label=f"TRT {args.precision.upper()}",
                  categories=args.categories, max_seqs=args.max_seqs,
                  checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
