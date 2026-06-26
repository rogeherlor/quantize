#!/usr/bin/env python3
"""TRT INT8 (ModelOpt PTQ) via the SLICED transformer chain — point #3 of the INT8 study.

The monolithic blocks export OOM-kills the 7.6 GB Orin, so we quantize once and export the
aggregator as N 8-pair slices (export_modelopt_ptq.py --chain), build each into an INT8 chain
engine (same recipe as the fp16 chain: onnx_to_fp16 → trtexec --stronglyTyped, with the
ModelOpt Q/DQ nodes making the QLAYERS INT8), and evaluate the hybrid chain (torch DINO embed
+ N INT8 chain engines + torch camera head) with the shared CO3D pose-AUC harness.

Compare the resulting AUC@30 to the fp16 chain (eval_results_chain_f10_c8.json, 0.653) to read
the INT8 quantization quality — identical pipeline + build recipe, only the Q/DQ differ.

Usage (inside the Jetson container):
  python scripts/bench_trt_modelopt_chain.py --algorithm max --categories apple --max_seqs 2
  python scripts/bench_trt_modelopt_chain.py --algorithm smoothquant --weight_per_tensor
"""
import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import _trt_bench_common as C

TOTAL_PAIRS = 24


def _trtexec():
    for c in ("trtexec", "/usr/src/tensorrt/bin/trtexec"):
        if os.path.exists(c) or subprocess.run(["bash", "-lc", f"command -v {c}"],
                                               capture_output=True).returncode == 0:
            return c
    raise SystemExit("trtexec not found")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=C.DEFAULT_CKPT)
    p.add_argument("--algorithm", choices=("max", "mse", "smoothquant"), default="max")
    p.add_argument("--weight_per_channel", dest="weight_per_channel", action="store_true", default=True)
    p.add_argument("--weight_per_tensor", dest="weight_per_channel", action="store_false")
    p.add_argument("--chunk_pairs", type=int, default=8)
    p.add_argument("--frames", type=int, default=C.DEFAULT_FRAMES)
    p.add_argument("--height", type=int, default=C.DEFAULT_H,
                   help="engine input height (350 default; 518 recovers the resolution-lost AUC)")
    p.add_argument("--width", type=int, default=C.DEFAULT_W)
    p.add_argument("--categories", default=None)
    p.add_argument("--max_seqs", type=int, default=None)
    p.add_argument("--calib_seqs", type=int, default=2)
    p.add_argument("--calib_frames", type=int, default=2)
    p.add_argument("--gpu_calib", action="store_true",
                   help="calibrate on GPU (faster but risks OOM-crashing the Tegra); default is "
                        "CPU/fp32 calibration, which is crash-safe on the 7.6 GB board")
    args = p.parse_args()

    res_sfx = "" if (args.height, args.width) == (C.DEFAULT_H, C.DEFAULT_W) else f"_{args.height}x{args.width}"
    tag = f"chain_modelopt_{args.algorithm}_{'pc' if args.weight_per_channel else 'pt'}{res_sfx}"
    base = C.onnx_path(tag)                       # <root>/onnx/<tag>/<tag>.onnx
    bbase, ext = os.path.splitext(base)
    starts = list(range(0, TOTAL_PAIRS, args.chunk_pairs))
    slice_onnx = [f"{bbase}_s{s}{ext}" for s in starts]
    engines = [C.engine_path(f"{tag}_s{s}") for s in starts]

    # 1. Sliced quantized ONNX (subprocess: heavy model load → memory-isolated from build/eval).
    if not all(os.path.exists(o) for o in slice_onnx):
        cmd = [sys.executable, os.path.join(_HERE, "export_modelopt_ptq.py"), "--chain",
               "--algorithm", args.algorithm,
               "--weight_per_channel" if args.weight_per_channel else "--weight_per_tensor",
               "--chunk_pairs", str(args.chunk_pairs), "--frames", str(args.frames),
               "--height", str(args.height), "--width", str(args.width),
               "--calib_seqs", str(args.calib_seqs), "--calib_frames", str(args.calib_frames),
               "--checkpoint", args.checkpoint, "--out", base]
        if not args.gpu_calib:
            cmd.append("--cpu")   # crash-safe CPU calibration (default)
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False")
        print(f"=== export sliced INT8 ONNX ===\n{' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=_ROOT, env=env)

    # 2. Build each slice INT8 engine — same recipe as the fp16 chain (onnx_to_fp16 →
    #    trtexec --stronglyTyped); the ModelOpt Q/DQ make the QLAYERS INT8.
    trtexec = _trtexec()
    for o, eng, s in zip(slice_onnx, engines, starts):
        if os.path.exists(eng):
            continue
        fp16 = f"{bbase}_s{s}_fp16{ext}"
        if not os.path.exists(fp16):
            subprocess.run([sys.executable, os.path.join(_HERE, "onnx_to_fp16.py"),
                            "--in", o, "--out", fp16], check=True, cwd=_ROOT)
        print(f"=== build INT8 chain engine s{s} → {eng} ===", flush=True)
        subprocess.run([trtexec, f"--onnx={fp16}", f"--saveEngine={eng}", "--stronglyTyped",
                        "--maxAuxStreams=0", "--memPoolSize=workspace:512",
                        "--builderOptimizationLevel=3", "--skipInference",
                        f"--timingCacheFile={C.timing_cache_path()}"], check=True, cwd=_ROOT)

    # 3. Evaluate the hybrid INT8 chain (torch DINO + INT8 chain engines + torch camera head).
    results = C.results_path("trt_modelopt_" + ('pc' if args.weight_per_channel else 'pt')
                             + "_" + args.algorithm + res_sfx)
    C.eval_chain(engines, results,
                 label=f"TRT INT8 chain ModelOpt ({args.algorithm},"
                       f"{'pc' if args.weight_per_channel else 'pt'}{res_sfx})",
                 categories=args.categories, max_seqs=args.max_seqs,
                 checkpoint=args.checkpoint, height=args.height, width=args.width,
                 num_frames=args.frames)


if __name__ == "__main__":
    main()
