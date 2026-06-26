#!/usr/bin/env python3
"""Config #3 — the WHOLE pose path in TensorRT (no PyTorch compute).

The monolithic full-model engine cannot build on the Orin Nano, so the path is a chain
of single-I/O engines that each build under the contiguous-memory ceiling:

    images → dino_0 → dino_1 → chain_s0 → chain_s8 → chain_s16 → camera_head → pose_enc

This script just evaluates that chain (the engines are built separately via trtexec —
see scripts/trt_chain_build.sh for the transformer chunks and the DINO/camera-head
build commands in JETSON_PLAN.md §10). All six engines must already exist in
deploy/artifacts/engines/.

Usage (inside the Jetson container):
  python scripts/bench_trt_fulltrt.py --categories apple --max_seqs 2
  python scripts/bench_trt_fulltrt.py
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

import _trt_bench_common as C

# Execution order: DINO (2 chunks) → transformer (3 chunks) → camera head.
DEFAULT_CHAIN = [
    "dino_0", "dino_1",
    "chain_f10_c8_s0", "chain_f10_c8_s8", "chain_f10_c8_s16",
    "camera_head",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engines", nargs="+", default=None,
                   help="engine basenames (no .trt) in execution order; "
                        f"default: {' '.join(DEFAULT_CHAIN)}")
    p.add_argument("--results", default=None)
    p.add_argument("--label", default="TRT all-engine FP16 (#3)")
    p.add_argument("--categories", default=None)
    p.add_argument("--max_seqs", type=int, default=None)
    p.add_argument("--num_frames", type=int, default=C.DEFAULT_FRAMES)
    p.add_argument("--height", type=int, default=C.DEFAULT_H)
    p.add_argument("--width", type=int, default=C.DEFAULT_W)
    args = p.parse_args()

    names = args.engines or DEFAULT_CHAIN
    engine_paths = [C.engine_path(n) for n in names]
    missing = [p for p in engine_paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("missing engines (build them first):\n  " + "\n  ".join(missing))
    results = args.results or C.results_path("trt_fulltrt_fp16")

    C.eval_all_engines(engine_paths, results, label=args.label,
                       categories=args.categories, max_seqs=args.max_seqs,
                       height=args.height, width=args.width, num_frames=args.num_frames)


if __name__ == "__main__":
    main()
