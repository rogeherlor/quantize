#!/usr/bin/env bash
# Build the N-engine CHAIN that covers the full 24-pair aggregator transformer in
# chunks of CHUNK pairs (each chunk's fused weight block is small enough to build on
# the Nano; the whole 24-pair stack's ~1.15 GB block is not). For each chunk it
# exports the FP16 slice → fp16-converts → builds with trtexec (--skipInference; the
# chain is timed end-to-end at eval). Prints the engine list + the eval command.
#
# Usage (inside the Jetson container):
#   bash scripts/trt_chain_build.sh 10 8     # 10 frames, 8 pairs/chunk → 3 engines
set -uo pipefail
FRAMES="${1:-10}"
CHUNK="${2:-8}"
TOTAL=24
ART=deploy/artifacts
TRTEXEC="$(command -v trtexec || true)"
[ -z "$TRTEXEC" ] && [ -x /usr/src/tensorrt/bin/trtexec ] && TRTEXEC=/usr/src/tensorrt/bin/trtexec
mkdir -p "$ART/engines" "$ART/profiles" "$ART/timing_cache"

engines=()
start=0
while [ "$start" -lt "$TOTAL" ]; do
  pairs=$CHUNK
  [ $((start + pairs)) -gt $TOTAL ] && pairs=$((TOTAL - start))
  end=$((start + pairs))
  islast=""; [ "$end" -eq "$TOTAL" ] && islast="--slice_is_last"
  name="chain_f${FRAMES}_c${CHUNK}_s${start}"
  onnx="$ART/onnx/$name/$name.onnx"
  onnx16="$ART/onnx/${name}_fp16/${name}_fp16.onnx"
  eng="$ART/engines/${name}.trt"
  mkdir -p "$(dirname "$onnx")" "$(dirname "$onnx16")"
  echo "=== chunk start=$start pairs=$pairs ${islast:-(middle)} ==="
  if [ ! -f "$onnx" ]; then
    python scripts/trt_export_pose.py --precision slice --frames "$FRAMES" \
      --slice_start "$start" --slice_pairs "$pairs" $islast --out "$onnx"
  fi
  [ -f "$onnx16" ] || python scripts/onnx_to_fp16.py --in "$onnx" --out "$onnx16"
  if "$TRTEXEC" --onnx="$onnx16" --saveEngine="$eng" --stronglyTyped --maxAuxStreams=0 \
       --memPoolSize=workspace:512 --builderOptimizationLevel=3 --skipInference \
       --timingCacheFile="$ART/timing_cache/orin_sm87.cache" \
       > "$ART/profiles/${name}_build.log" 2>&1; then
    echo "built $eng ($(du -h "$eng" | cut -f1))"
  else
    echo "BUILD FAILED for $name — see $ART/profiles/${name}_build.log"
    exit 1
  fi
  engines+=("$eng")
  start=$end
done

# Comma-separated quoted python list of engine paths, in pair order.
pylist=""; for e in "${engines[@]}"; do pylist="${pylist}'${e}',"; done
echo ""
echo "Chain built: ${#engines[@]} engines"
printf '  %s\n' "${engines[@]}"
echo ""
echo "Evaluate the chain (smoke: 1 category, 2 seqs):"
cat <<EOF
python -c "import sys; sys.path.insert(0,'scripts'); import _trt_bench_common as C; \\
C.eval_chain([${pylist}], C.results_path('chain_f${FRAMES}_c${CHUNK}'), \\
label='TRT FP16 chain (${CHUNK}/chunk)', num_frames=${FRAMES}, categories='apple', max_seqs=2)"
EOF
