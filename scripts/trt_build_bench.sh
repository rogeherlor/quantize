#!/usr/bin/env bash
# Stage 1 — build + benchmark TensorRT engines for the VGGT pose path with
# trtexec (no Python tensorrt binding needed). Run inside the Jetson container.
#
# Usage:
#   bash scripts/trt_build_bench.sh deploy/artifacts/onnx/full_fp16/full_fp16.onnx fp16
#   bash scripts/trt_build_bench.sh deploy/artifacts/onnx/full_int8_lsq/full_int8_lsq.onnx int8
#
# The ONNX graphs are fixed-shape (S=10, 518x350), so no shape flags are needed.
# Engines go to deploy/artifacts/engines/, logs + layer profiles to .../profiles/,
# and a persistent Orin timing cache (.../timing_cache/) speeds up rebuilds.
set -uo pipefail

ART="deploy/artifacts"
ONNX="${1:-$ART/onnx/full_fp16/full_fp16.onnx}"
MODE="${2:-fp16}"            # fp16 | int8
NAME="$(basename "${ONNX%.onnx}")"
mkdir -p "$ART/engines" "$ART/profiles" "$ART/timing_cache"
ENGINE="$ART/engines/${NAME}.trt"
PROFILE="$ART/profiles/${NAME}_layers.json"
TIMING_CACHE="$ART/timing_cache/orin_sm87.cache"
LOG="$ART/profiles/${NAME}_trtexec.log"

# Locate trtexec (PATH, or the standard JetPack location).
TRTEXEC="$(command -v trtexec || true)"
[ -z "$TRTEXEC" ] && [ -x /usr/src/tensorrt/bin/trtexec ] && TRTEXEC=/usr/src/tensorrt/bin/trtexec
if [ -z "$TRTEXEC" ]; then
  echo "ERROR: trtexec not found (looked on PATH and /usr/src/tensorrt/bin)." >&2
  exit 1
fi
echo "Using trtexec: $TRTEXEC"
echo "ONNX:   $ONNX"
echo "Engine: $ENGINE  (mode=$MODE)"

if [ ! -f "$ONNX" ]; then
  echo "ERROR: $ONNX not found. Run scripts/trt_export_pose.py first." >&2
  exit 1
fi

PREC_FLAGS="--fp16"                       # fp16 fallback always on
[ "$MODE" = "int8" ] && PREC_FLAGS="--fp16 --int8"

# 4 GB build workspace — Orin Nano has 7.8 GB unified; leave room for the OS.
# --timingCacheFile persists Orin kernel-tactic timings (fast rebuilds).
# --dumpProfile + --exportProfile give per-layer timing (profilingVerbosity=detailed).
"$TRTEXEC" \
  --onnx="$ONNX" \
  --saveEngine="$ENGINE" \
  $PREC_FLAGS \
  --memPoolSize=workspace:4096 \
  --builderOptimizationLevel=3 \
  --timingCacheFile="$TIMING_CACHE" \
  --iterations=50 \
  --avgRuns=20 \
  --warmUp=2000 \
  --noDataTransfers \
  --dumpProfile \
  --exportProfile="$PROFILE" \
  --profilingVerbosity=detailed \
  --verbose 2>&1 | tee "$LOG"

echo ""
echo "=== Latency summary (grep from log) ==="
grep -E "GPU Compute Time|Throughput|Latency" "$LOG" | tail -20
echo "Per-layer profile: $PROFILE"
echo ""
echo "Engine: $ENGINE ($(du -h "$ENGINE" 2>/dev/null | cut -f1))"
