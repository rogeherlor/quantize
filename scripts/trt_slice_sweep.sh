#!/usr/bin/env bash
# Sweep aggregator sub-engine sizes to find the LARGEST that builds on the Nano and
# measure per-engine GPU-compute latency. For each PAIRS value it: exports a FP16 slice
# (pairs [0:PAIRS)), fp16-converts, then builds+benchmarks with trtexec, and records
# whether it built, the engine size, and the median GPU compute time.
#
# Usage (inside the Jetson container):
#   bash scripts/trt_slice_sweep.sh 6 "1 2 4 6 8 12"   # frames=6, pair counts to try
#   bash scripts/trt_slice_sweep.sh 10 "1 2 4 6"       # toward the real S=10
set -uo pipefail
FRAMES="${1:-6}"
PAIRS_LIST="${2:-1 2 4 6 8 12}"
ART=deploy/artifacts
TRTEXEC="$(command -v trtexec || true)"
[ -z "$TRTEXEC" ] && [ -x /usr/src/tensorrt/bin/trtexec ] && TRTEXEC=/usr/src/tensorrt/bin/trtexec
mkdir -p "$ART/engines" "$ART/profiles" "$ART/timing_cache"

echo ""
echo "Sweep: frames=$FRAMES  pairs={$PAIRS_LIST}"
echo "| frames | pairs | weights | built | engine_MB | GPU_ms (median) |"
echo "|--------|-------|---------|-------|-----------|-----------------|"
for P in $PAIRS_LIST; do
  name="slice_f${FRAMES}_p${P}"
  onnx="$ART/onnx/$name/$name.onnx"
  onnx16="$ART/onnx/${name}_fp16/${name}_fp16.onnx"
  eng="$ART/engines/${name}.trt"
  log="$ART/profiles/${name}_trtexec.log"
  mkdir -p "$(dirname "$onnx")" "$(dirname "$onnx16")"

  if [ ! -f "$onnx" ]; then
    python scripts/trt_export_pose.py --precision slice --frames "$FRAMES" \
        --slice_start 0 --slice_pairs "$P" --slice_is_last --out "$onnx" \
        > "$ART/profiles/${name}_export.log" 2>&1
  fi
  [ -f "$onnx16" ] || python scripts/onnx_to_fp16.py --in "$onnx" --out "$onnx16" \
        > /dev/null 2>&1

  "$TRTEXEC" --onnx="$onnx16" --saveEngine="$eng" --stronglyTyped \
     --maxAuxStreams=0 --memPoolSize=workspace:512 --builderOptimizationLevel=2 \
     --timingCacheFile="$ART/timing_cache/orin_sm87.cache" > "$log" 2>&1

  wmb=$(( P * 48 ))   # ~48 MB per pair (2 blocks) in fp16 weights
  if grep -q "&&&& PASSED" "$log"; then
    mb=$(grep -oE "Created engine with size: [0-9.]+" "$log" | grep -oE "[0-9.]+" | head -1)
    ms=$(grep -oE "GPU Compute Time:.*median = [0-9.]+" "$log" | grep -oE "median = [0-9.]+" | grep -oE "[0-9.]+")
    echo "| $FRAMES | $P | ~${wmb}MB | yes | ${mb:-?} | ${ms:-?} |"
  else
    echo "| $FRAMES | $P | ~${wmb}MB | NO | - | - |"
  fi
done
echo ""
echo "Per-pair logs in $ART/profiles/slice_f${FRAMES}_p*_trtexec.log"
echo "The largest 'yes' pairs value → chunk size; N sub-engines = ceil(24 / chunk)."
