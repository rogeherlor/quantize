# Deployment Strategy: LSQ Quantized VGGT on Jetson Orin Nano

## Hardware Target
- **Board**: Jetson Orin Nano (8 GB)  
- **GPU**: Ampere sm87 — 1024 CUDA cores  
- **DLA**: 2× NVDLA (supports INT8)  
- **Stack**: JetPack 6.1 · CUDA 12.6 · cuDNN 9.x · TensorRT 10.3

---

## Phase Overview

| Phase | Precision       | TRT Support    | Effort    | Status  |
|-------|-----------------|----------------|-----------|---------|
| 1     | **W8A8 INT8**   | Native         | 3–5 days  | TODO    |
| 2     | **W4A16 INT4w** | Experimental   | 2–3 days  | TODO    |
| 3     | **W4A4 INT4**   | Custom kernel  | 1–2 weeks | TODO    |

---

## Quantizer Strategy → TRT Export Mapping

| Training quantizer | Export approach | Notes |
|--------------------|-----------------|-------|
| `LSQ_quantizer`    | Per-tensor `QuantizeLinear`/`DequantizeLinear` (Q/DQ) | Scale from `w_Qparms['scale']`, `x_Qparms['scale']` |
| `LSQC_quantizer`   | Per-channel Q/DQ on weights, per-tensor on activations | `scale.shape = (out, 1)` → ONNX axis=0 |
| `LSQS_quantizer`   | Fold smooth factor into weights before export, then per-tensor Q/DQ | `w_export = w * exp(smooth_log_scale)`, zero overhead at runtime |
| `LSQH_quantizer`   | Pre-apply Hadamard to weights (`rotated_weight`), then per-tensor Q/DQ | Rotation baked in; no runtime overhead |
| `LSQBH_quantizer`  | Pre-apply block Hadamard to weights, then per-tensor Q/DQ | Same as LSQH but per-block |
| `LSQHD_quantizer`  | Apply diagonal sign flip + Hadamard to weights, then per-tensor Q/DQ | Fold `d_vec` and `H` into weights |

**Key export principle**: at `model.eval()`, `_forward_common` already applies smooth folding and Hadamard caching via `module.rotated_weight`. Run one dummy forward before exporting to populate these caches, then export — the rotated/smoothed weights will be frozen in the ONNX graph.

---

## Phase 1: W8A8 via TensorRT INT8

### Goal
Take a QAT checkpoint (W8A8 LSQ) and produce a `.trt` engine that runs real INT8 tensor core operations on the Jetson GPU.

### Export pipeline
```
QAT checkpoint (QLinear with learned scales)
    ↓  deploy/export/onnx_exporter.py
ONNX opset 17 with Q/DQ nodes (QuantizeLinear + DequantizeLinear)
    ↓  deploy/export/trt_builder.py  (or trtexec)
Serialized .trt engine with INT8 precision
    ↓  deploy/runtime/trt_inference.py
Real inference on Jetson
```

### Key files
- `deploy/export/onnx_exporter.py` — loads checkpoint, patches QLinear to emit Q/DQ
- `deploy/export/trt_builder.py` — builds TRT engine; sets optimization profiles for variable N frames
- `deploy/runtime/trt_inference.py` — runs `.trt` engine
- `deploy/runtime/benchmark.py` — measures latency/FPS vs FP32 baseline

### Expected result
- ~2–3× latency speedup over FP32  
- <2% depth mAE degradation vs FP32 baseline

### Checklist
- [ ] Install JetPack 6.1 + `pip install nvidia-modelopt[torch] onnxruntime-gpu`
- [ ] Export VGGT QAT checkpoint to ONNX with Q/DQ nodes
- [ ] Validate ONNX output vs PyTorch (max abs error < 1% of value range)
- [ ] Build TRT INT8 engine with optimization profiles
- [ ] Run `trt_inference.py` and compare depth maps vs FP32 baseline
- [ ] Run `benchmark.py` — record latency (ms/frame) and FPS

---

## Phase 2: W4A16 via ModelOpt + TRT

### Goal
Compress weights to INT4 while keeping activations at FP16. Beneficial for VGGT because transformer inference is memory-bandwidth bound.

### Path
```
FP32 or QAT checkpoint
    ↓  nvidia-modelopt: mtq.quantize(model, mtq.INT4_WEIGHT_ONLY_CFG)
ModelOpt-quantized model
    ↓  mto.export_to_onnx(model)
ONNX with INT4 weight Q/DQ + FP16 activations
    ↓  trtexec --int4 (TRT 10.x experimental)
.trt engine
```

### Note on compatibility
ModelOpt applies its own quantization, independent of the LSQ training. For Phase 2 the strategy is to start from the **FP32 checkpoint** and apply ModelOpt INT4 PTQ, rather than trying to adapt the custom LSQ W4A16 checkpoint. This is the pragmatic demo path.

### Checklist
- [ ] `pip install nvidia-modelopt[torch]` on Jetson
- [ ] Run `mtq.quantize(vggt_fp32, mtq.INT4_WEIGHT_ONLY_CFG)` with CO3D calibration data
- [ ] Export to ONNX via ModelOpt exporter
- [ ] Build TRT engine with `--int4` flag
- [ ] Benchmark: compare W4A16 vs W8A8 latency

---

## Phase 3: W4A4 via Custom CUTLASS Kernel

### Goal
Deploy the custom W4A4 LSQ-trained model with real INT4 × INT4 matrix multiplications, using CUTLASS on sm87.

### Why custom kernel is needed
- TensorRT 10.x does not expose W4A4 (INT4 activations) through its standard Q/DQ path
- Ampere (sm87) supports INT4 tensor ops via CUTLASS 3.x with interleaved 4-bit layout
- Expected gain over W8A8 on Ampere: ~1.3–1.5× (not 2×; full 2× requires Hopper/H100)

### Implementation path
1. Write `deploy/kernels/int4_gemm.cu` using CUTLASS `cutlass::gemm::device::Gemm<int4b_t, int4b_t, int32_t>` with interleaved layout
2. Build as PyTorch CUDA extension via `deploy/kernels/setup.py`
3. Write `deploy/plugins/int4_plugin.py` wrapping the kernel as a TRT `IPluginV3`
4. In `deploy/export/onnx_exporter.py`: for W4A4 models, insert the custom plugin node instead of standard Q/DQ

### Checklist
- [ ] Build CUTLASS INT4 GEMM for sm87 (validate on Jetson)
- [ ] Benchmark raw kernel throughput vs `torch._int_mm` (INT8 baseline)
- [ ] Write TRT plugin wrapper (`IPluginV3`)
- [ ] End-to-end: W4A4 LSQ checkpoint → TRT with custom plugin → compare vs W8A8

---

## Key Decisions Log

| Decision | Rationale |
|----------|-----------|
| W8A8 first, not W4A4 | Fully supported by TRT; W4A4 is marginal speedup on Ampere vs complexity of custom kernel |
| ModelOpt for Phase 2 (not custom LSQ W4A16) | ModelOpt handles the entire TRT-export flow; avoids mapping custom LSQ scales to TRT INT4 format |
| PyTorch CUDA extension for W4A4 kernel | Easier to prototype than TRT IPluginV3; switch to plugin later if graph optimization is needed |
| Fix batch=1, dynamic N frames (1–8) in TRT profiles | Demo target; Orin Nano has 8 GB so multi-batch isn't the bottleneck |
| Use `rotated_weight` cache for Hadamard export | `_forward_common` already caches at eval; one dummy forward populates it before ONNX export |

---

## Environment Setup

```bash
# On Jetson (JetPack 6.1)
sudo apt-get install -y tensorrt python3-libnvinfer-dev

# Python packages (use Jetson-specific torch wheel)
pip install nvidia-modelopt[torch]
pip install onnxruntime-gpu onnx onnxsim
pip install polygraphy --extra-index-url https://pypi.ngc.nvidia.com

# Validate TRT install
python -c "import tensorrt; print(tensorrt.__version__)"
```

---

## Results Log

| Date | Model | Precision | Latency (ms/frame) | FPS | Depth mAE | Notes |
|------|-------|-----------|-------------------|-----|-----------|-------|
| -    | VGGT  | FP32      | -                 | -   | baseline  | -     |
| -    | VGGT  | W8A8 TRT  | -                 | -   | -         | Phase 1 |
| -    | VGGT  | W4A16 TRT | -                 | -   | -         | Phase 2 |
| -    | VGGT  | W4A4 CUTLASS | -              | -   | -         | Phase 3 |
