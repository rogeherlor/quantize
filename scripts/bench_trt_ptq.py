#!/usr/bin/env python3
"""Approach 4 — TensorRT INT8 with automatic PTQ calibration.

Builds an INT8 engine from the plain (non-QDQ) pose ONNX using TRT's own entropy
calibration on real CO3D frames, but CONSTRAINED to the same QLAYERS Linears as the
LSQ approach (every other MatMul/Conv is pinned to FP16). This isolates the only
variable vs bench_trt_lsq.py: scale source (entropy calibration vs learned LSQ).

Usage (inside the Jetson container):
  python scripts/bench_trt_ptq.py --categories apple --max_seqs 2
  python scripts/bench_trt_ptq.py
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
    p.add_argument("--calib_batches", type=int, default=64)
    p.add_argument("--workspace_gb", type=float, default=4.0)
    args = p.parse_args()

    onnx = args.onnx or C.onnx_path("pose_baseline")
    engine = args.engine or C.engine_path("pose_int8_ptq")
    results = args.results or C.results_path("trt_int8_ptq")

    # 1. Plain BLOCKS ONNX (no Q/DQ, input = tokens) — TRT will calibrate it.
    C.ensure_baseline_onnx(onnx, checkpoint=args.checkpoint)

    # 2. Restrict INT8 to the SAME 12 QLAYERS blocks as LSQ (strip the
    #    "aggregator." prefix so the substring matches the ONNX/TRT layer names).
    from src.run_realquant import QLAYERS
    restrict = [q.split("aggregator.")[-1] for q in QLAYERS]

    # 3. Token calibrator: load the torch DINO embed, run it on real CO3D frames to
    #    produce the engine's `tokens` input, then free it before the TRT build.
    import gc
    import torch
    sys.path.insert(0, _HERE)
    import eval_co3d_realquant as erq
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emodel = erq._load_model(args.checkpoint, torch.float16, device)
    emodel.depth_head = emodel.point_head = emodel.track_head = None
    agg = emodel.aggregator

    def embed_fn(images):
        with torch.inference_mode():
            tokens, *_ = agg.embed_tokens(images.to(device, torch.float16))
        return tokens

    from deploy.export.calibration import CO3DCalibrator
    calib = CO3DCalibrator(embed_fn=embed_fn, num_frames=C.DEFAULT_FRAMES,
                           height=C.DEFAULT_H, width=C.DEFAULT_W,
                           num_batches=args.calib_batches)
    del embed_fn, agg, emodel
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # 4. Build INT8 engine (automatic calibration, QLAYERS-only).
    from deploy.export.trt_builder import build_engine
    build_engine(onnx, engine, precision="int8",
                 workspace_gb=args.workspace_gb, calibrator=calib,
                 restrict_int8_to=restrict, timing_cache=C.timing_cache_path())

    # 5. Evaluate → comparable JSON.
    C.eval_engine(engine, results, label="TRT INT8 (auto PTQ)",
                  categories=args.categories, max_seqs=args.max_seqs,
                  checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
