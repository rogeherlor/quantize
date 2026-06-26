# VGGT Inference on Jetson Orin Nano Super — Implementation Plan

## 1. Objective

Run VGGT (1.2 B-parameter visual geometry transformer) inference on the
Jetson Orin Nano Super using FP16 weights. The target is correct, end-to-end
pose/depth prediction without crashing the device, with acceptable latency for
offline evaluation (not hard real-time).

---

## 2. Hardware Inventory

### 2.1 Board

| Property | Value |
|---|---|
| Board | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super |
| JetPack | R36 rev 4.7 (JetPack 6.2, kernel 5.15.148-tegra) |
| CPU | 6-core ARMv8 (Cortex-A78AE) @ ~1.5 GHz, aarch64 |
| GPU | Orin iGPU (Ampere, compute capability 8.7) |
| CUDA cores | ~1024 |
| Tensor cores | Yes (3rd-gen, supports FP16/INT8) |
| FP16 throughput | ~1.3 TFLOPS (theoretical) |

### 2.2 Memory (unified — CPU and GPU share the same physical pool)

| Pool | Total | Used at idle | Available |
|---|---|---|---|
| Unified RAM | 7.80 GB | ~3.7 GB | ~3.6 GB |
| Swap (zram ×6) | 3.8 GB | ~3 MB | ~3.8 GB |

**Critical**: there is no separate VRAM. Every byte allocated by CUDA comes out
of the same 7.80 GB that the OS, Python process, and dataloader workers also
use. `nvidia-smi` reports `total_memory = 7619 MB` — this is the entire system
RAM, not a dedicated GPU budget.

### 2.3 Storage

| Device | Path | Total | Free | Interface | Notes |
|---|---|---|---|---|---|
| eMMC (internal) | `/` | 59.5 GB | **6.1 GB** | eMMC 5.1 | 89 % full — near limit |
| NVMe M.2 SSD | `/ssd` | 931.5 GB | **795 GB** | PCIe NVMe | ext4, ~2–4 GB/s seq read |

**eMMC is critically full.** All model checkpoints, datasets, and Docker images
must live on `/ssd`. Do not write large files to `/`.

Current `/ssd` usage:
- `/ssd/co3d/` — CO3D dataset (co3d sequences for evaluation)
- `/ssd/docker/` — Docker storage root (container images and layers)

### 2.4 Software Stack (inside container `quantize-jetson`)

| Component | Version |
|---|---|
| PyTorch | 2.7.0 |
| CUDA toolkit | 12.6 |
| Python | 3.10 |
| xformers | **Not installed** (falls back to PyTorch SDPA — acceptable) |
| cuDNN | via JetPack 6.2 |

---

## 3. VGGT Architecture — Inference Memory Analysis

### 3.1 Model Dimensions

| Parameter | Value |
|---|---|
| Total parameters | 1.2 B |
| Backbone | ViT-L variant (Aggregator) |
| embed_dim | 1024 |
| depth | 24 transformer blocks |
| num_heads | 16 (head_dim = 64) |
| MLP ratio | 4.0 |
| Image size | 518 × 518 px |
| Patch size | 14 × 14 px |
| Patches per frame | 37 × 37 = 1 369 |
| Register tokens | 4 |
| Tokens per frame | 1 373 |
| Tokens for 10 frames | **13 730** |

### 3.2 Attention Strategy

The model uses `F.scaled_dot_product_attention` (PyTorch native SDPA), which on
compute 8.7 (Ampere) automatically dispatches to the Flash Attention 2 kernel.
This means attention does **not** materialize the full N×N matrix in memory;
peak attention memory is O(N), not O(N²). xformers is absent but not required.

### 3.3 Memory Budget — FP16 Inference (10 frames, no_grad)

| Item | Size | Notes |
|---|---|---|
| FP16 weights on CUDA | **2.40 GB** | 1.2 B × 2 bytes |
| Peak activations (global attn block) | **0.14 GB** | QKV + SDPA + FFN, sequential |
| CUDA runtime + Python + workers | **1.50 GB** | empirical OS baseline |
| CUDA allocator fragmentation reserve | **0.30 GB** | safety margin |
| **Total estimated peak** | **4.34 GB** | |
| Available unified RAM | 7.80 GB | |
| **Headroom** | **3.46 GB** | comfortable |

FP16 inference **fits in unified memory** with ~3.5 GB to spare.
The SSD is not needed as a VRAM overflow device during inference.

### 3.4 Checkpoint Loading — The Danger Zone

The original checkpoint is FP32 (4.80 GB). Loading it naively while the FP16
model is already on CUDA creates a dangerous peak:

| Load strategy | Peak RAM | Safe? |
|---|---|---|
| `torch.load(path, map_location='cpu')` — naive | 4.80 + 2.40 = **7.20 GB** | Risky (only 0.6 GB headroom) |
| `torch.load(path, map_location='cpu', mmap=True)` | ~3.12 GB peak (demand-paged) | Safe |
| Pre-converted FP16 checkpoint, `map_location='cuda'` | ~2.52 GB peak | **Ideal** |

**Action**: always load checkpoints with `mmap=True`, or better, save a FP16
checkpoint once and load that directly to CUDA from then on.

---

## 4. Scope

### In scope
- FP16 inference of VGGT on Jetson Orin Nano Super inside Docker container
- CO3D pose evaluation (`run_evaluation_vggt`) as the primary benchmark
- Standalone single-image / small-batch inference via `infer.py`
- Memory-safe checkpoint loading from `/ssd`
- Quantization experiments (INT8, INT4) using the existing QAT/PTQ pipeline,
  with evaluation on the same CO3D benchmark

### Out of scope
- Training or fine-tuning on-device (7.8 GB unified memory is insufficient
  for the backward pass and optimizer state of a 1.2 B-parameter model)
- Real-time / streaming inference (latency not yet characterized)
- Multi-device or multi-Jetson inference
- FP32 baseline on device (4.80 GB weights alone would leave < 3 GB for
  activations and runtime — borderline and fragile)

---

## 5. Identified Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| OOM during `model.to(device)` — CPU→CUDA copy peaks at 4.8 GB | High | **Fixed**: `torch.set_default_device('cuda')` in `_setup_components` creates model directly on CUDA |
| OOM during naive FP32 checkpoint load (7.2 GB peak) | High | Use `mmap=True`; long-term: save FP16 checkpoint |
| eMMC full (6.1 GB free) — OS may refuse writes | High | All data, checkpoints, and logs must be on `/ssd` |
| NCCL P2P probe fails on Jetson (single GPU) | Medium | **Fixed**: backend forced to `gloo` in Jetson mode |
| NumPy 1.x/2.x ABI mismatch warning | Low | Warning only; does not affect correctness |
| Long inference latency (~1–5 s/pass at 1.3 TFLOPS FP16) | Medium | Accept for offline eval; profile before optimizing |
| Quantization accuracy regression | Medium | Always compare AUC@30 against FP16 baseline before shipping INT8 |

---

## 6. Storage Strategy

```
/ssd/
├── co3d/
│   ├── dataset/          # raw CO3D sequences
│   └── preprocessed_dataset/   # preprocessed annotations (.jgz)
├── docker/               # Docker storage root (images + layers)
├── model_zoo/
│   └── vggt/
│       ├── vggt_1B_commercial.pt      # original FP32 checkpoint (~4.8 GB)
│       └── vggt_1B_fp16.pt            # [TODO] exported FP16 checkpoint (~2.4 GB)
└── logs/                 # TensorBoard logs, evaluation outputs
```

Symlink or configure `data3/rogelio/` to point into `/ssd` if not already done.

---

## 7. Implementation Roadmap

### Phase 0 — Unblock baseline inference (current state)

- [x] Set `backend=gloo` for Jetson (no NCCL P2P)
- [x] Set `model_dtype=float16` via `--jetson` flag
- [x] Create model directly on CUDA via `torch.set_default_device` in
      `_setup_components` — eliminates the CPU→CUDA double-allocation OOM
- [ ] Verify baseline FP16 run completes and logs AUC metrics (this is
      the first thing to confirm after the current crash is fixed)

### Phase 1 — Harden memory handling

- [ ] Add `mmap=True` to all `torch.load` calls in the codebase
- [ ] Export and save a FP16 checkpoint to `/ssd/model_zoo/vggt/vggt_1B_fp16.pt`
- [ ] Update `infer.py` to instantiate the model in FP16 directly on CUDA
      (currently creates in FP32, then moves — same OOM risk as the Trainer had)
- [ ] Add a memory monitor (`torch.cuda.memory_summary()`) at key checkpoints
      to measure actual peaks vs estimates

### Phase 2 — Characterize performance

- [ ] Measure actual inference latency per sequence (10 frames)
- [ ] Measure memory peak empirically with `torch.cuda.max_memory_allocated()`
- [ ] Sweep `num_frames` (3, 5, 7, 10) to understand the latency/accuracy
      trade-off on-device
- [ ] Profile with `torch.profiler` to find the top time consumers

### Phase 3 — Quantization (INT8 / INT4)

- [ ] Run INT8 PTQ (existing `gptq`/`rtn` pipeline) and measure AUC regression
- [ ] Validate that INT8 weights fit: 1.2 B × 1 byte = 1.2 GB (huge headroom)
- [ ] If latency is too high in FP16, try TensorRT export via `torch2trt` or
      `trtexec` — Ampere tensor cores can deliver 2× FP16 improvement with TRT
      engine optimizations

### Phase 4 — Packaging

- [ ] Finalize `Dockerfile.jetson` for reproducible deployment
- [ ] Document exact `docker run` command and volume mounts
- [ ] Add a minimal `infer.py` smoke test that runs in < 60 s end-to-end

---

## 8. Key Commands

```bash
# Run FP16 baseline (inside container)
python main.py \
  --mode QAT \
  --num_bits 32 \
  --w_quantizer FP --x_quantizer FP \
  --w_first_last_quantizer FP --x_first_last_quantizer FP \
  --initializer None \
  --train_id fp16_baseline \
  --lr 0.0 --coeff_qparm_lr 0.0 --weight_decay 0.0 --qparm_wd 0.0 \
  --jetson

# Export FP16 checkpoint (run once after baseline loads)
python3 -c "
import torch
sd = torch.load('/ssd/model_zoo/vggt/vggt_1B_commercial.pt',
                map_location='cpu', mmap=True)
sd_fp16 = {k: v.half() for k, v in sd.items()}
torch.save(sd_fp16, '/ssd/model_zoo/vggt/vggt_1B_fp16.pt')
print('saved')
"

# Standalone inference
python infer.py

# Monitor memory during a run
watch -n 1 "free -h && echo '---' && cat /proc/meminfo | grep MemAvailable"
```

---

## 9. Why the SSD Is Not a VRAM Substitute

PyTorch's NVMe-based VRAM offloading (e.g., DeepSpeed ZeRO-Infinity) works by
streaming parameter shards from disk during the forward pass. This requires
custom kernels and significantly increases latency. On a Jetson with ~2–4 GB/s
NVMe throughput and only ~1.3 TFLOPS compute, the NVMe bandwidth would become
the bottleneck long before compute does. The math:

- FP16 weights: 2.4 GB; at 3 GB/s NVMe → 0.8 s just to read weights per pass
- FP16 compute at 50 % efficiency: 0.65 TFLOPS; forward pass ≈ 0.6 s

Disk-offloaded inference would be 2–3× slower than memory-resident inference
with no accuracy benefit. **The SSD should be used for storage, not as overflow
VRAM.** We have enough unified memory for FP16 inference without it.

The SSD **is** valuable for:
1. Storing the CO3D dataset and model checkpoints (eMMC is full)
2. Fast dataset streaming during evaluation (NVMe >> eMMC)
3. Docker image and layer storage

---

## 10. TensorRT Acceleration — Findings & Working Approach

### 10.1 What is possible

**The full 24-block aggregator transformer CAN run in TensorRT FP16 on the Orin Nano
at the original 10-frame CO3D benchmark — as a CHAIN of 3 sub-engines.** DINO
patch-embed + camera head stay in PyTorch FP16 around the chain (hybrid pipeline). This
unblocks evaluating the standard 10-frame benchmark with a TensorRT-accelerated
transformer; frame count is **not** the constraint (see below).

What does *not* work on this board:
- A single **full-model** engine (`images → pose_enc`, DINO + aggregator + camera head):
  fails — the DINO weights inflate the build past memory and the camera-head fusion
  (`ForeignNode[...camera_head/Concat]`) cannot be placed (`Error 10`).
- A single **whole-transformer** engine (all 24 pairs in one engine): fails to *build*
  (not run) for the reason in §10.2.

### 10.2 The build-memory wall (why one engine fails)

The blocker is **engine build time**, not inference. TensorRT 10.3 on the 7.6 GB board:

| Cause | Detail |
|---|---|
| Builder kernel library | TRT reserves **~3.0–3.8 GB** GPU *before the network is parsed* |
| Fused weight constant | TRT's Myelin compiler fuses the transformer into one node and packs **all** its weights into **one ~1.15 GB contiguous** allocation (≈ 48 MB × 24 pairs) |
| Frame-independent | The 1.15 GB block is **identical at 6 and 10 frames** → it is *weights*, not activations (activations did shrink 209→126 MB with fewer frames, but were never the blocker) |
| NvMap fragmentation | Autotuning allocates/frees transient buffers, so even with several GB free *in aggregate*, NvMap cannot find a **contiguous** 1.15 GB block → `NvMapMemAllocInternalTagged ... error 12` → CUDA OOM |
| No fusion knob | TensorRT exposes no way to disable the Myelin fusion or split the fused weight block *within one engine* (`builderOptimizationLevel=0` does not un-fuse it) |

Levers tried and exhausted on the single engine (all failed): FP16 weight pre-conversion,
`--stronglyTyped`, weight streaming (`allowWeightStreaming` @ 0 % budget), `maxAuxStreams=0`,
workspace 256 MB–1 GB, builder optimisation levels 0–3, headless reboot (freed ~1.5 GB),
fewer frames (10→6). The contiguous block only shrank ~1.15 GB → ~0.8 GB and never built.

### 10.3 What works: depth-split sub-engine chain

Split the transformer's **depth** — the 24 alternating frame/global attention "pairs" —
into chunks. Each chunk is its **own engine** = its own Myelin region = its own **small**
weight block, which allocates contiguously even when NvMap is fragmented. (A separate
engine is the only unit TensorRT lets you control; there is no "one engine, separate
weight blocks".) The chunks are chained at inference; the running token state
`(B*S, P, C)` passes from one engine to the next.

**Measured build envelope (per-chunk weight block sets the limit):**

| pairs/chunk | weights | builds? |
|---|---|---|
| 1–8 | ~48 MB × pairs (≤ 384 MB) | **yes** |
| 12 | ~576 MB | **no** |

→ contiguous ceiling ≈ 400–500 MB. **Chunk = 8 pairs → N = ⌈24/8⌉ = 3 engines** cover the
whole transformer (use 6 pairs → N = 4 for more margin). Same limit at 6 and 10 frames.

**Per-chunk build recipe (lean, on-device):**
1. Export the FP16 slice — `trt_export_pose.py --precision slice` (no quantisation → no
   calibration forward → **no export OOM**; tiny weights).
2. fp16-convert the ONNX — `scripts/onnx_to_fp16.py` (onnx-only; halves the build-time
   weight footprint; offline-container safe — no `onnxconverter-common` needed).
3. Build with `trtexec --stronglyTyped --maxAuxStreams=0 --memPoolSize=workspace:512`
   (strongly-typed avoids the heavy fp32 attention tactics; lean flags minimise the peak).

**Runtime:** `TRTChainedPoseModel` (deploy/runtime/trt_inference.py) = PyTorch DINO embed →
N chunk engines in order → PyTorch camera head. `eval_chain()` (scripts/_trt_bench_common.py)
runs it through the standard CO3D pose-AUC harness.

### 10.4 Measured numbers (FP16, single-slice sweep)

| frames | pairs | engine MB | GPU compute (median) |
|---|---|---|---|
| 1 | 1 | 48.7 | 11.7 ms |
| 6 | 1 | 48.9 | 79.6 ms |
| 6 | 8 | 389 | 623 ms |
| 10 | 1 | 49.1 | 156 ms |
| 10 | 8 | 391 | 1213 ms |

Latency is **linear in pairs** (~152 ms/pair @ 10 frames, ~80 ms @ 6) → the full 24-pair
transformer ≈ **3.6 s** of TRT-FP16 GPU compute at 10 frames (3 chunks × ~1.21 s). The
6→10-frame jump is super-linear (global attention is O(tokens²)).

### 10.5 How to run

```bash
# build the 3-engine chain (export → fp16-convert → build each chunk; slow, run detached)
nohup bash scripts/trt_chain_build.sh 10 8 \
  > deploy/artifacts/profiles/chain_build.log 2>&1 &      # frames=10, 8 pairs/chunk

# evaluate end-to-end on CO3D (prints AUC@30/15/5/3, p50/p90/p99 latency, total engine MB)
python -c "import sys; sys.path.insert(0,'scripts'); import _trt_bench_common as C; \
C.eval_chain(['deploy/artifacts/engines/chain_f10_c8_s0.trt', \
'deploy/artifacts/engines/chain_f10_c8_s8.trt', \
'deploy/artifacts/engines/chain_f10_c8_s16.trt'], \
C.results_path('chain_f10_c8'), label='TRT FP16 chain (8/chunk)', \
num_frames=10, categories='apple', max_seqs=2)"
```
Artifacts land under `deploy/artifacts/` (→ `/ssd/profiles/quantize`): `onnx/`, `engines/`,
`profiles/`, `results/`, `timing_cache/`.

### 10.6 Constraints & next step (INT8)

- **Engines are not portable** across GPU architectures — always build on the Nano (a
  4070/desktop GPU cannot cross-build for it; an AGX/NX Orin could, same SM 8.7).
- **Offline container** (no internet): `nvidia-modelopt` and `onnxconverter-common` cannot
  be pip-installed; fp16 conversion therefore uses the onnx-only `scripts/onnx_to_fp16.py`.
- **FP16 is not a large latency win** over the eager PyTorch hybrid (both use Flash-Attention
  on Ampere). The real target is **INT8** (memory + tensor-core compute).
- **INT8 must be quantised per-chunk.** Calibrating all 48 blocks at once OOMs at export
  (fake-quant ~doubles activation memory → SSH freeze). Quantising **8 blocks per chunk**
  keeps calibration within budget — so the depth-split *also* unblocks INT8, which is the
  next step once the FP16 chain is validated end-to-end.

### 10.7 First end-to-end execution — runtime fixes & result

The 3-engine FP16 chain **ran end-to-end at the original 10 frames** (`bench_trt_all`'s
`eval_chain`). Getting from "3 engines built" to "runs the CO3D harness" required a chain
of *runtime* memory/correctness fixes (all in `deploy/runtime/trt_inference.py`):

| Problem | Symptom | Fix |
|---|---|---|
| Loading 3 engines back-to-back fragments NvMap | engine 3 deserialize OOM on a 403 MB contiguous block | **Defrag**: reserve+free one big contiguous block before loading engines |
| Engines + full-model load co-resident | `373 MB free` → Jetson `NVML_SUCCESS` assert | **Reorder**: load torch parts first (peak 2.4 GB → drop to 0.99 GB), *then* the engines |
| `embed_tokens` needs RoPE | `pos = pos + 1` on `None` | **Keep `rope`** (tiny) when dropping the block weights |
| Several engines pin I/O buffers | `cuMemHostAlloc failed` (pinned host pool exhausted) | **Pageable** host buffers (`np.empty`) instead of `pagelocked_empty` |
| CO3D loader gives 4-D images | `interpolate` spatial-dim mismatch | **Unsqueeze** `(S,C,H,W)→(1,S,C,H,W)` in the wrappers |

**First measured result (apple, 2 seqs, identical seeded frames as the baseline):**

| Configuration | AUC@30 | Latency (s/seq) | Peak CUDA |
|---|---|---|---|
| PyTorch FP16 (with depth head) | 0.8126 | 15.2 | 4094 MB |
| **TRT FP16 chain (3×8 pairs), pose-only** | **0.6530** | **6.1** | **1114 MB** |

- **Memory: a clear win** — 1.1 GB peak vs 4.0 GB (the transformer's weights live in the
  engines; only DINO embed + camera head + rope stay in torch, ~0.99 GB).
- **Latency** is lower but not a clean comparison — the PyTorch baseline ran the DPT depth
  head; the chain is pose-only. A pose-only PyTorch baseline is needed for a fair speedup.
- **Accuracy regressed (0.81 → 0.65)** on *identical* frames. DINO embed + camera head are
  the **unmodified** originals and the camera head correctly reads only the last token
  (`aggregated_tokens_list[-1]`), so this is **not** a structural bug. It is **precision**:
  the engines are `--stronglyTyped` (all-fp16, forced to keep the build within memory),
  whereas PyTorch's fp16 LayerNorm/softmax/RoPE accumulate internally in fp32; over 48
  blocks the error compounds. **Open fix:** a precision-aware fp16 export that keeps the
  normalisation/RoPE ops in fp32 while the matmul weights stay fp16 (so the build still
  fits) — expected to recover accuracy toward the 0.81 baseline.

### 10.8 Full all-TRT path (config #3) — runs, but needs *opposite* memory handling

Config #3 puts the **entire** pose path in TensorRT — no torch compute — as a 6-engine
chain (`images → dino_0 → dino_1 → chain_s0 → chain_s8 → chain_s16 → camera_head →
pose_enc`). The monolithic full-model engine cannot build (DINO's 605 MB weight const +
the unplaceable camera-head fusion, §10.2), so DINO is split into 2 chunks (~300 MB each)
and the camera head is its own engine; all six are `--stronglyTyped` fp16. Runtime:
`TRTAllEngineModel` + `eval_all_engines` + `scripts/bench_trt_fulltrt.py`.

**Why the memory handling is the *inverse* of the hybrid chain (§10.7).** The difference is
**who owns the big weights**:
- **#2 hybrid:** torch owns the un-quantized weights (~0.99 GB DINO embed + camera head) in
  its flexible *pooled caching* allocator; only 3 transformer engines (~1.17 GB) are in TRT.
  Torch grabs one large CUDA segment up front while memory is free, so later allocations
  reuse it — memory "just works."
- **#3 all-TRT:** the **whole 1.2 B model lives in TRT** (~2.2 GB weights across 6 engines +
  0.5 GB scratch), allocated by the rigid CUDA **driver** allocator; torch owns ~0 MB on the
  GPU. That inversion breaks three things the hybrid never hits:

| Problem (unique to all-TRT) | Cause | Fix (`deploy/runtime/trt_inference.py`) |
|---|---|---|
| OOM loading the 4th–6th engine on a ~0.4 GB contiguous block | 6 contiguous weight consts (vs 3) into fragmented unified memory | **Defrag before deserialize** + run on a **freshly dropped page cache** (`echo 3 > drop_caches`) |
| OOM creating the 5th execution context (403 MB) | 6 contexts × ~0.5 GB private scratch = ~3 GB | **One shared scratch buffer** (max engine size, 536 MB) bound to all 6 contexts — legal because they run strictly sequentially |
| `NVML_SUCCESS == r` assert on a *tiny* torch CUDA alloc | torch owns no warm segment → its expandable-segments allocator must **grow** → queries `nvmlDeviceGetMemoryInfo`, **unsupported on Tegra** | (a) defrag via **pycuda/driver** not torch (a multi-GB torch alloc/free poisons torch's NVML accounting); (b) keep `forward` **entirely host-side** in torch (host arrays straight between engines); (c) launch with **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`** (uses `cudaMemGetInfo`, which works on Tegra) |

(Plus a buffer-reuse bug: `infer()` compared a 1-D host buffer's `.shape` against the multi-dim
input shape, so it re-allocated — and leaked — a device buffer every call; fixed to compare
element counts.)

**Measured result (apple, 2 seqs, identical seeded frames):**

| Configuration | AUC@30 | Latency (s/seq) | Resident | Robustness |
|---|---|---|---|---|
| #2 hybrid TRT chain (torch DINO+head) | **0.653** | **6.2** | 1.17 GB TRT + 0.99 GB torch | works as-is |
| #3 all-TRT (6 engines) | 0.175 | 9.7 | 2.17 GB TRT (`Model weights: 0 MB`) | needs cache-drop + `expandable_segments:False` |

**Conclusion — the two memory strategies are not interchangeable, and the hybrid wins on
this board.** Each strategy is *forced* by its architecture (you can't pool TRT weights in
torch when there is no torch model, and the driver-level handling is pointless when torch
already manages the weights), so neither is "better in both cases." But comparing the
*architectures*, **#2 is currently better on every axis on the Nano**: accuracy
(0.653 vs 0.175), latency (6.2 vs 9.7 s — #3 is *slower* because of 6 host round-trips), and
robustness. #3's only intrinsic edge is "self-contained engines, no torch at runtime"; on a
7.6 GB Tegra board that does not pay off. #3 stands as proof that full-TRT is *feasible*, not
as the deployable config. **Open items for #3:** (a) the accuracy regression is larger than
#2's precision floor (0.175 vs 0.653 on identical data → a real fidelity loss in `dino_0/1`
or `camera_head`, to localize with a DINO numdiff: TRT `dino_0→dino_1` "tokens" vs torch
`embed_tokens`); (b) device-to-device inter-engine handoff to remove the host bounces.
