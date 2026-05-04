#!/usr/bin/env bash
# One-shot: QAT checkpoint → ONNX → TRT engine
#
# Usage:
#   ./deploy/scripts/build_engine.sh \
#       --checkpoint model_zoo/pytorchcv/vggt_w8a8_lsq.pth \
#       --precision  int8 \
#       --num_frames 4
#
# Environment:
#   Run this on the Jetson Orin Nano (TRT engine must be built on target device).
#   JetPack 6.1 + CUDA 12.6 + TensorRT 10.3 required.

set -euo pipefail

CHECKPOINT=""
PRECISION="int8"
NUM_BITS=8
X_QUANTIZER="LSQ_quantizer"
W_QUANTIZER="LSQ_quantizer"
NUM_FRAMES=4
IMAGE_SIZE=224
MIN_FRAMES=1
OPT_FRAMES=4
MAX_FRAMES=8
OUT_DIR="deploy/engines"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)   CHECKPOINT="$2"; shift 2 ;;
        --precision)    PRECISION="$2";  shift 2 ;;
        --num_frames)   NUM_FRAMES="$2"; OPT_FRAMES="$2"; shift 2 ;;
        --num_bits)     NUM_BITS="$2";   shift 2 ;;
        --x_quantizer) X_QUANTIZER="$2"; shift 2 ;;
        --w_quantizer) W_QUANTIZER="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT" ]]; then
    echo "Error: --checkpoint is required"
    exit 1
fi

ONNX_PATH="${OUT_DIR}/vggt_${PRECISION}.onnx"
TRT_PATH="${OUT_DIR}/vggt_${PRECISION}.trt"

echo "=== Step 1: Export QAT checkpoint → ONNX ==="
python -m deploy.export.onnx_exporter \
    --checkpoint   "$CHECKPOINT" \
    --output       "$ONNX_PATH"  \
    --num_bits     "$NUM_BITS"   \
    --x_quantizer  "$X_QUANTIZER" \
    --w_quantizer  "$W_QUANTIZER" \
    --num_frames   "$NUM_FRAMES"

echo ""
echo "=== Step 2: Build TRT engine ==="
python -m deploy.export.trt_builder \
    --onnx        "$ONNX_PATH"   \
    --output      "$TRT_PATH"    \
    --precision   "$PRECISION"   \
    --min_frames  "$MIN_FRAMES"  \
    --opt_frames  "$OPT_FRAMES"  \
    --max_frames  "$MAX_FRAMES"  \
    --image_size  "$IMAGE_SIZE"

echo ""
echo "=== Done ==="
echo "Engine: $TRT_PATH"
echo "Size:   $(du -sh "$TRT_PATH" | cut -f1)"
