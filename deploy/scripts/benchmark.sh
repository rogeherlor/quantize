#!/usr/bin/env bash
# Run the benchmark suite on Jetson.
#
# Usage:
#   ./deploy/scripts/benchmark.sh \
#       --checkpoint model_zoo/pytorchcv/vggt_w8a8_lsq.pth \
#       --trt_engine deploy/engines/vggt_int8.trt \
#       --num_frames 4
#
# Also captures tegrastats power/temperature during benchmark.
# Output written to deploy/engines/benchmark_<timestamp>.log

set -euo pipefail

CHECKPOINT=""
TRT_ENGINE=""
NUM_FRAMES=4
IMAGE_SIZE=224
WARMUP=10
RUNS=50
LOG_DIR="deploy/engines"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/benchmark_${TIMESTAMP}.log"
NUM_BITS=8
X_QUANTIZER="LSQ_quantizer"
W_QUANTIZER="LSQ_quantizer"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)   CHECKPOINT="$2"; shift 2 ;;
        --trt_engine)   TRT_ENGINE="$2"; shift 2 ;;
        --num_frames)   NUM_FRAMES="$2"; shift 2 ;;
        --warmup)       WARMUP="$2";     shift 2 ;;
        --runs)         RUNS="$2";       shift 2 ;;
        --num_bits)     NUM_BITS="$2";   shift 2 ;;
        --x_quantizer) X_QUANTIZER="$2"; shift 2 ;;
        --w_quantizer) W_QUANTIZER="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

echo "=== VGGT Benchmark: $(date) ===" | tee "$LOG_FILE"
echo "Checkpoint: ${CHECKPOINT:-none}" | tee -a "$LOG_FILE"
echo "TRT engine: ${TRT_ENGINE:-none}" | tee -a "$LOG_FILE"
echo "Frames: $NUM_FRAMES, Warmup: $WARMUP, Runs: $RUNS" | tee -a "$LOG_FILE"

# Start tegrastats in background (Jetson power/thermal monitor)
if command -v tegrastats &> /dev/null; then
    echo "Starting tegrastats..." | tee -a "$LOG_FILE"
    tegrastats --interval 1000 >> "${LOG_FILE}.tegrastats" &
    TEGRASTATS_PID=$!
fi

# Run Python benchmark
python -m deploy.runtime.benchmark \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
    ${TRT_ENGINE:+--trt_engine "$TRT_ENGINE"} \
    --num_bits     "$NUM_BITS"     \
    --x_quantizer  "$X_QUANTIZER"  \
    --w_quantizer  "$W_QUANTIZER"  \
    --num_frames   "$NUM_FRAMES"   \
    --warmup_runs  "$WARMUP"       \
    --bench_runs   "$RUNS"         \
    2>&1 | tee -a "$LOG_FILE"

# Stop tegrastats
if [[ -n "${TEGRASTATS_PID:-}" ]]; then
    kill "$TEGRASTATS_PID" 2>/dev/null || true
    echo "Tegrastats log: ${LOG_FILE}.tegrastats" | tee -a "$LOG_FILE"
fi

echo ""
echo "Benchmark log: $LOG_FILE"
