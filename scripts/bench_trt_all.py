#!/usr/bin/env python3
"""Run the three comparable TensorRT BLOCKS engines on the Jetson, then aggregate.

  fp16      TRT FP16 blocks engine (no quantization)        — bench_trt_baseline.py
  lsq       + QLAYERS INT8 via learned LSQ Q/DQ             — bench_trt_lsq.py
  modelopt  + QLAYERS INT8 via NVIDIA ModelOpt PTQ Q/DQ     — bench_trt_modelopt.py

All three share the SAME blocks graph (tokens → last_tokens): the ENTIRE transformer
(aggregator) runs in TensorRT — FP16 everywhere except the 12 QLAYERS at INT8 — while
the DINO patch-embed and camera head run in PyTorch FP16 around it (the hybrid
pipeline). They differ only in the QLAYERS' precision / INT8 scale source.

(The full `images → pose_enc` engine is intentionally NOT built here: it does not fit
the Orin Nano's 7.4 GB at build time. See scripts/TRT_BENCHMARK.md.)

Each approach runs as its OWN subprocess so the heavy export/build/eval stages don't
share unified memory — critical on the 7.4 GB board. ONNX/engines/results land under
deploy/artifacts/ and reuse the persistent Orin timing cache.

Usage (inside the Jetson container):
  python scripts/bench_trt_all.py --which fp16 --categories apple --max_seqs 2
  python scripts/bench_trt_all.py --which all
"""
import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")

# variant → (script, extra args). calib_* only apply to the quantized approaches.
APPROACHES = {
    "fp16":     ("bench_trt_baseline.py", ["--precision", "fp16"], False),
    "lsq":      ("bench_trt_lsq.py",      [],                      True),
    "modelopt": ("bench_trt_modelopt.py", [],                      True),
}


def run_approach(which, args):
    script, extra, takes_calib = APPROACHES[which]
    cmd = [sys.executable, os.path.join(_HERE, script), "--checkpoint", args.checkpoint] + extra
    if args.categories:
        cmd += ["--categories", args.categories]
    if args.max_seqs is not None:
        cmd += ["--max_seqs", str(args.max_seqs)]
    if takes_calib:
        cmd += ["--calib_seqs", str(args.calib_seqs), "--calib_frames", str(args.calib_frames)]
    print(f"\n=== [{which}] {' '.join(cmd)} ===", flush=True)
    subprocess.run(cmd, check=True, cwd=_ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", choices=["fp16", "lsq", "modelopt", "all"], default="all")
    p.add_argument("--checkpoint", default="data3/rogelio/model_zoo/vggt/vggt_1B_commercial.pt")
    p.add_argument("--categories", default=None)
    p.add_argument("--max_seqs", type=int, default=None)
    p.add_argument("--calib_seqs", type=int, default=2)
    p.add_argument("--calib_frames", type=int, default=4)
    p.add_argument("--no_compare", action="store_true", help="skip the comparison table at the end")
    args = p.parse_args()

    todo = ["fp16", "lsq", "modelopt"] if args.which == "all" else [args.which]
    for which in todo:
        run_approach(which, args)

    if not args.no_compare:
        print("\n=== comparison ===", flush=True)
        subprocess.run([sys.executable, os.path.join(_HERE, "compare_results.py")],
                       check=True, cwd=_ROOT)


if __name__ == "__main__":
    main()
