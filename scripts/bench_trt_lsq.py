#!/usr/bin/env python3
"""Approach 5 — LSQ w8a8 → QDQ → TensorRT INT8 (the headline "our approach").

Exports the pose graph with explicit Q/DQ nodes carrying the LEARNED LSQ scales
(only on the 12 QLAYERS), builds an INT8 engine from it (no calibrator — scales come
from the QDQ nodes), and evaluates with the shared pose-AUC harness.

Usage (inside the Jetson container):
  python scripts/bench_trt_lsq.py --categories apple --max_seqs 2
  python scripts/bench_trt_lsq.py
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
    p.add_argument("--workspace_gb", type=float, default=4.0)
    args = p.parse_args()

    onnx = args.onnx or C.onnx_path("pose_w8a8_lsq")
    engine = args.engine or C.engine_path("pose_w8a8_lsq")
    results = args.results or C.results_path("trt_w8a8_lsq")

    # 1. Export the QDQ ONNX (calibrates LSQ scales on GPU, exports on CPU fp32).
    C.ensure_w8a8_onnx(onnx, checkpoint=args.checkpoint,
                       calib_seqs=args.calib_seqs, calib_frames=args.calib_frames)

    # 2. Build INT8 engine — scales come from the Q/DQ nodes, no calibrator.
    from deploy.export.trt_builder import build_engine
    build_engine(onnx, engine, precision="int8",
                 workspace_gb=args.workspace_gb, timing_cache=C.timing_cache_path())

    # 3. Evaluate → comparable JSON.
    C.eval_engine(engine, results, label="TRT INT8 (LSQ w8a8)",
                  categories=args.categories, max_seqs=args.max_seqs,
                  checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
