#!/usr/bin/env python3
"""Aggregate every approach's eval_results*.json into one comparison table.

Reads the JSON written by run_evaluation_vggt (PyTorch rows) and augmented by the
bench_trt_* scripts (TRT rows), and emits output/comparison.md + .csv plus a stdout
table with: memory, end-to-end latency, pure GPU-compute latency (TRT), pose AUC@30,
speedup vs the PyTorch FP16 baseline, and memory reduction vs that baseline.

Usage:
  python scripts/compare_results.py
  python scripts/compare_results.py --results_dir output --out output/comparison.md
"""
import argparse
import csv
import glob
import json
import os

# filename → (display label, precision) in preferred display order.
# The headline trio: FP16 blocks vs INT8-LSQ vs INT8-ModelOpt (same blocks engine,
# whole transformer in TRT; only the QLAYERS' scale source differs).
KNOWN = [
    ("eval_results.json",                       "PyTorch FP16 (baseline)",   "fp16"),
    ("eval_results_trt_fp16.json",              "TRT FP16 (blocks)",         "fp16"),
    ("eval_results_trt_w8a8_lsq.json",          "TRT INT8 (LSQ)",            "int8"),
    ("eval_results_trt_modelopt.json",          "TRT INT8 (ModelOpt)",       "int8"),
    ("eval_results_trt_fp32.json",              "TRT FP32 (blocks)",         "fp32"),
    ("eval_results_trt_int8_ptq.json",          "TRT INT8 (auto PTQ)",       "int8"),
    ("eval_results_w8a8.json",                  "PyTorch w8a8 (_int_mm)",    "w8a8"),
    ("eval_results_w8a4.json",                  "PyTorch w8a4 (_int_mm)",    "w8a4"),
    ("eval_results_w4a4.json",                  "PyTorch w4a4 (_int_mm)",    "w4a4"),
]

BASELINE_FILE = "eval_results.json"  # PyTorch FP16 — reference for speedup/memory


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _mem_mb(res):
    """Engine size for TRT rows, weight memory for PyTorch rows."""
    if res.get("engine_mb") is not None:
        return res["engine_mb"]
    return res.get("weight_memory_mb")


def _lat_ms(res, stat="mean"):
    lat = (res.get("seq_latency_s") or {}).get(stat)
    return None if lat is None else lat * 1000.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="deploy/artifacts/results")
    p.add_argument("--out", default="deploy/artifacts/results/comparison.md")
    args = p.parse_args()

    # Collect known files first (ordered), then any other eval_results*.json.
    rows = []
    seen = set()
    for fname, label, prec in KNOWN:
        path = os.path.join(args.results_dir, fname)
        res = _load(path)
        if res is not None:
            rows.append((label, prec, res))
            seen.add(os.path.abspath(path))
    for path in sorted(glob.glob(os.path.join(args.results_dir, "eval_results*.json"))):
        if os.path.abspath(path) in seen:
            continue
        res = _load(path)
        if res is not None:
            rows.append((os.path.basename(path), "?", res))

    if not rows:
        print(f"No eval_results*.json found in {args.results_dir}/")
        return

    base = _load(os.path.join(args.results_dir, BASELINE_FILE))
    base_lat = _lat_ms(base) if base else None
    base_mem = _mem_mb(base) if base else None

    table = []  # dict rows for csv + md
    for label, prec, res in rows:
        mem = _mem_mb(res)
        lat = _lat_ms(res)
        lat_p90 = _lat_ms(res, "p90")
        lat_p99 = _lat_ms(res, "p99")
        auc = (res.get("mean_auc") or {}).get("auc30")
        gpu = res.get("trtexec_gpu_ms")
        speedup = (base_lat / lat) if (base_lat and lat) else None
        mem_red = (base_mem / mem) if (base_mem and mem) else None
        table.append({
            "approach": label,
            "precision": prec,
            "memory_mb": mem,
            "latency_ms_per_seq": None if lat is None else round(lat, 1),
            "lat_p90_ms": None if lat_p90 is None else round(lat_p90, 1),
            "lat_p99_ms": None if lat_p99 is None else round(lat_p99, 1),
            "gpu_compute_ms": gpu,
            "auc30": None if auc is None else round(auc, 4),
            "speedup_vs_fp16": None if speedup is None else round(speedup, 2),
            "mem_reduction_vs_fp16": None if mem_red is None else round(mem_red, 2),
        })

    # ── stdout + markdown ────────────────────────────────────────────────────
    cols = ["approach", "precision", "memory_mb", "latency_ms_per_seq",
            "lat_p90_ms", "lat_p99_ms", "gpu_compute_ms", "auc30",
            "speedup_vs_fp16", "mem_reduction_vs_fp16"]
    hdr = ["Approach", "Prec", "Mem (MB)", "Latency (ms/seq)",
           "p90 (ms)", "p99 (ms)", "GPU (ms)", "AUC@30", "Speedup×", "Mem×"]

    def fmt(v):
        return "—" if v is None else str(v)

    md = ["# VGGT TensorRT comparison (pose-only, Jetson Orin Nano)\n",
          "Memory = TRT engine size for TRT rows, weight memory for PyTorch rows. "
          "Speedup/Mem× are vs the PyTorch FP16 baseline. GPU (ms) is trtexec "
          "`--noDataTransfers` pure compute (TRT rows only).\n",
          "| " + " | ".join(hdr) + " |",
          "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in table:
        md.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    md_text = "\n".join(md) + "\n"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(md_text)
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)

    print(md_text)
    print(f"Wrote {args.out} and {csv_path}")


if __name__ == "__main__":
    main()
