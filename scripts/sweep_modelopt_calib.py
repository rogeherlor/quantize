#!/usr/bin/env python3
"""Optimise the INT8 quantization: sweep ModelOpt calibration configs and pick the best AUC.

For each (algorithm × weight granularity) it runs scripts/eval_modelopt_torch.py as its OWN
subprocess (memory isolation — each loads the model fresh; critical on the 7.6 GB Orin), reads
the AUC from the per-config results JSON, and prints a ranked table. The winning config is the
one to feed the TensorRT INT8 engine (scripts/bench_trt_modelopt.py --algorithm ... ).

Calibration methods are the ModelOpt 0.44 ones that suit W8A8 (max / mse / smoothquant).

Usage (inside the Jetson container):
  python scripts/sweep_modelopt_calib.py --categories apple --max_seqs 2
  python scripts/sweep_modelopt_calib.py --algorithms max mse --no_per_tensor
"""
import argparse
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
RESULTS_DIR = os.path.join(_ROOT, "deploy", "artifacts", "results")


def _auc30(results_path):
    try:
        with open(results_path) as f:
            return (json.load(f).get("mean_auc") or {}).get("auc30")
    except (OSError, json.JSONDecodeError):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--algorithms", nargs="+", default=["max", "mse", "smoothquant"])
    p.add_argument("--no_per_tensor", action="store_true",
                   help="only sweep per-channel weights (skip per-tensor)")
    p.add_argument("--categories", default="apple")
    p.add_argument("--max_seqs", type=int, default=2)
    p.add_argument("--calib_seqs", type=int, default=2)
    p.add_argument("--calib_frames", type=int, default=4)
    p.add_argument("--act_pool_mb", type=int, default=2600)
    args = p.parse_args()

    granularities = [True] if args.no_per_tensor else [True, False]
    rows = []
    for alg in args.algorithms:
        for per_channel in granularities:
            tag = f"{alg}_{'pc' if per_channel else 'pt'}"
            results = os.path.join(RESULTS_DIR, f"eval_results_modelopt_torch_{tag}.json")
            cmd = [sys.executable, os.path.join(_HERE, "eval_modelopt_torch.py"),
                   "--algorithm", alg,
                   "--weight_per_channel" if per_channel else "--weight_per_tensor",
                   "--calib_seqs", str(args.calib_seqs),
                   "--calib_frames", str(args.calib_frames),
                   "--act_pool_mb", str(args.act_pool_mb),
                   "--results", results]
            if args.checkpoint:
                cmd += ["--checkpoint", args.checkpoint]
            if args.categories:
                cmd += ["--categories", args.categories]
            if args.max_seqs is not None:
                cmd += ["--max_seqs", str(args.max_seqs)]
            print(f"\n=== [{tag}] {' '.join(cmd)} ===", flush=True)
            rc = subprocess.run(cmd, cwd=_ROOT).returncode
            auc = _auc30(results) if rc == 0 else None
            rows.append((tag, alg, per_channel, auc, rc))

    rows.sort(key=lambda r: (r[3] is None, -(r[3] or 0)))
    print("\n================ ModelOpt calibration sweep (AUC@30) ================")
    print(f"{'config':18s} {'algorithm':12s} {'weights':11s} {'AUC@30':>8s}  status")
    for tag, alg, pc, auc, rc in rows:
        a = "—" if auc is None else f"{auc:.4f}"
        st = "ok" if rc == 0 else f"FAILED(rc={rc})"
        print(f"{tag:18s} {alg:12s} {'per-channel' if pc else 'per-tensor':11s} {a:>8s}  {st}")
    best = next((r for r in rows if r[3] is not None), None)
    if best:
        print(f"\nBest: {best[0]} (AUC@30={best[3]:.4f}) → build TRT INT8 with:\n"
              f"  python scripts/bench_trt_modelopt.py --algorithm {best[1]} "
              f"{'--weight_per_channel' if best[2] else '--weight_per_tensor'}")
    else:
        print("\nNo config produced a result (all failed).")


if __name__ == "__main__":
    main()
