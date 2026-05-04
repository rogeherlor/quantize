/**
 * INT4 × INT4 GEMM kernel for Jetson Orin Nano (sm87, Ampere).
 *
 * Uses CUTLASS 3.x with interleaved 4-bit weight layout.
 * Intended as the backend for W4A4 real-integer inference.
 *
 * Phase 3 — implement after W8A8 TRT is working and W4A4 speedup is needed.
 *
 * Build via:
 *   python deploy/kernels/setup.py build_ext --inplace
 *
 * Prerequisites:
 *   - CUTLASS 3.x cloned to /opt/cutlass (or adjust include path below)
 *   - CUDA 12.6 + nvcc with sm87 target
 *   - torch >= 2.0 for pybind11 extension
 *
 * References:
 *   CUTLASS INT4 GEMM: examples/cute/tutorial/wgmma_sm90.cu
 *   Ampere INT4 layout: include/cutlass/layout/tensor_op_multiplicand_sm80.h
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

// ──────────────────────────────────────────────────────────────────────────────
// TODO: CUTLASS includes (uncomment when CUTLASS is installed)
// ──────────────────────────────────────────────────────────────────────────────
// #include "cutlass/cutlass.h"
// #include "cutlass/gemm/device/gemm.h"
// #include "cutlass/layout/tensor_op_multiplicand_sm80.h"
// #include "cutlass/epilogue/thread/linear_combination.h"

// ──────────────────────────────────────────────────────────────────────────────
// INT4 packing utilities (CPU-side, used by the PyTorch extension)
// ──────────────────────────────────────────────────────────────────────────────

/**
 * pack_int4_weights
 *
 * Converts a float32 weight tensor [out_features, in_features] (pre-quantized
 * to [-8, 7] integer range) into CUTLASS interleaved INT4 layout.
 *
 * Interleaved layout packs 2 INT4 values per byte:
 *   byte[i] = (w[2i] & 0xF) | ((w[2i+1] & 0xF) << 4)
 *
 * Args:
 *   w_int: INT8 tensor (values in [-8, 7]) of shape [out, in].
 * Returns:
 *   Packed UINT8 tensor of shape [out, in/2].
 */
torch::Tensor pack_int4_weights(torch::Tensor w_int) {
    TORCH_CHECK(w_int.dtype() == torch::kInt8, "Expected int8 input");
    TORCH_CHECK(w_int.dim() == 2, "Expected 2D weight tensor");
    TORCH_CHECK(w_int.size(1) % 2 == 0, "in_features must be even");

    auto out = w_int.size(0);
    auto in  = w_int.size(1);

    // Shift to unsigned range [0, 15]
    auto w_uint = (w_int + 8).to(torch::kUInt8);

    // Pack two columns into one byte
    auto w_low  = w_uint.slice(1, 0, in, 2);   // even columns
    auto w_high = w_uint.slice(1, 1, in, 2);   // odd columns
    return (w_high << 4) | w_low;              // [out, in/2]
}

/**
 * unpack_int4_weights
 *
 * Inverse of pack_int4_weights. Returns int8 tensor with values in [-8, 7].
 */
torch::Tensor unpack_int4_weights(torch::Tensor w_packed) {
    TORCH_CHECK(w_packed.dtype() == torch::kUInt8, "Expected uint8 packed tensor");
    auto w_low  = (w_packed & 0x0F).to(torch::kInt8) - 8;
    auto w_high = ((w_packed >> 4) & 0x0F).to(torch::kInt8) - 8;
    // Interleave back: [out, in/2] → [out, in]
    return torch::stack({w_low, w_high}, 2).flatten(1);
}

// ──────────────────────────────────────────────────────────────────────────────
// INT4 GEMM — placeholder for CUTLASS kernel
// ──────────────────────────────────────────────────────────────────────────────

/**
 * int4_gemm
 *
 * Computes: out = (x_int4 @ w_int4.T) * (x_scale * w_scale)
 *
 * Where x_int4 and w_int4 are quantized to 4-bit symmetric range [-8, 7].
 * The actual INT4 tensor-core operation is performed via CUTLASS; this
 * placeholder falls back to INT8 matmul (torch._int_mm) for prototyping.
 *
 * Args:
 *   x_int8:   INT8 activation tensor [M, K] (values clamped to [-8, 7]).
 *   w_packed: Packed INT4 weight tensor [N, K/2] (from pack_int4_weights).
 *   x_scale:  Scalar activation dequantization scale.
 *   w_scale:  Scalar weight dequantization scale (or [N,1] per-channel).
 *
 * Returns:
 *   Float32 output tensor [M, N].
 */
torch::Tensor int4_gemm(
    torch::Tensor x_int8,
    torch::Tensor w_packed,
    float x_scale,
    torch::Tensor w_scale
) {
    // Unpack weights to INT8 (CUTLASS INT4 kernel goes here later)
    auto w_int8 = unpack_int4_weights(w_packed);  // [N, K]

    // Fallback: use torch._int_mm (INT8 tensor core) while CUTLASS is not wired
    // TODO: replace with CUTLASS INT4 device::Gemm when available
    auto out_int32 = torch::_int_mm(x_int8, w_int8.t());  // [M, N]
    return out_int32.to(torch::kFloat32) * (x_scale * w_scale);
}

// ──────────────────────────────────────────────────────────────────────────────
// Python bindings
// ──────────────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_int4_weights",   &pack_int4_weights,   "Pack int8 → interleaved INT4 (2 per byte)");
    m.def("unpack_int4_weights", &unpack_int4_weights, "Unpack interleaved INT4 → int8");
    m.def("int4_gemm",           &int4_gemm,           "INT4 × INT4 GEMM (Ampere sm87)");
}
