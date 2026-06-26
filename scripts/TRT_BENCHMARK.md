# VGGT TensorRT Acceleration & Memory Comparison

How to run the per-approach benchmark suite and what it is measuring.

## Objective

Produce a **fair, apples-to-apples comparison** of real hardware acceleration and
memory reduction for VGGT inference on the **Jetson Orin Nano Super**, across several
quantization approaches. Each approach has its own script; all of them write the same
JSON schema, and `compare_results.py` merges them into one table
(`output/comparison.md`).

The comparison answers three questions:
1. Does **TensorRT** actually accelerate VGGT vs the PyTorch eager baseline?
2. Is our **LSQ** quantization (learned scales) better than **TensorRT's automatic
   PTQ** calibration?
3. What memory reduction does each precision buy?

### Two hard constraints (do not break these)

- **Identical quantized layer set in every approach.** Only the 12 `QLAYERS`
  aggregator blocks (their `attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2` Linears) are
  quantized; everything else stays FP16. The TRT automatic-PTQ build is explicitly
  constrained to the same layers (otherwise it would quantize the whole graph and the
  comparison would be meaningless).
- **Identical pose-only head config in every approach.** Camera head ON; depth, point,
  and track heads OFF. The point/track heads explode memory and can't export tractably,
  and pose AUC@30 only depends on `pose_enc`, so pose-only is both correct and
  memory-safe.

Because of these, the only variable between the INT8 rows is the **scale source**
(entropy calibration vs learned LSQ), which is exactly what we want to study.

### Hybrid engine: only the aggregator blocks are in TensorRT

The TRT engine covers **only the aggregator's frame/global block stack** — the heavy,
quantized region where the QLAYERS live. The pipeline at inference is:

```
images ──PyTorch──▶ Aggregator.embed_tokens (DINO patch-embed + token assembly)
       ──TensorRT─▶ blocks engine   (tokens [F,P,C] → last-block tokens [B,S,P,2C])
       ──PyTorch──▶ camera_head      (reads only the last block → pose_enc [B,S,9])
```

Why split here rather than export the whole pose path as one engine:
- The DINO embed and camera head are **never quantized** and are identical FP16 torch
  across every approach, so the engine = exactly the variable under study.
- The camera head consumes only `aggregated_tokens_list[-1]`, so the engine has a
  single clean output — no list-of-intermediates to plumb.
- A monolithic ~2.4 GB fp32 ONNX OOM-killed export validation on Orin; the blocks-only
  graph exports, validates (in a subprocess), and builds within Jetson memory.

Consequences baked into the numbers: the engine's `tokens` input means the **PTQ
calibrator feeds tokens** (it runs the torch DINO embed on real CO3D frames first), and
the **GPU (ms)** column measures the blocks only, while end-to-end latency includes the
torch DINO + head and the H2D/D2H token transfers.

> Engines/ONNX are large (block weights ≈ 2.4 GB fp32). On the Jetson, point
> `deploy/engines` at a roomy volume — e.g. `ln -s /ssd/.../engines deploy/engines` —
> so a >2 GB export never fills the 57 GB root.

## Headline comparison: three blocks engines (`bench_trt_all.py`)

`scripts/bench_trt_all.py` runs the three comparable engines and aggregates them. All
three share the **same blocks graph** (`tokens → last_tokens`): the **entire transformer
(aggregator) runs in TensorRT** — FP16 everywhere except the 12 QLAYERS at INT8 — while
DINO patch-embed + camera head run in PyTorch FP16 (the hybrid pipeline below). They
differ only in the QLAYERS' INT8 scale source:

| Variant | What | Script | Engine |
|---|---|---|---|
| `fp16` | blocks engine, no quantization | `bench_trt_baseline.py --precision fp16` | `pose_fp16.trt` |
| `lsq` | + QLAYERS INT8 via learned **LSQ** Q/DQ | `bench_trt_lsq.py` | `pose_w8a8_lsq.trt` |
| `modelopt` | + QLAYERS INT8 via **NVIDIA ModelOpt** PTQ Q/DQ | `bench_trt_modelopt.py` | `pose_modelopt.trt` |

```bash
# smoke: 1 category, 2 seqs (each approach runs as its own subprocess → memory-isolated)
python scripts/bench_trt_all.py --which fp16     --categories apple --max_seqs 2
python scripts/bench_trt_all.py --which lsq      --categories apple --max_seqs 2
python scripts/bench_trt_all.py --which modelopt --categories apple --max_seqs 2  # needs nvidia-modelopt
# full sweep + comparison table
python scripts/bench_trt_all.py --which all
```
The INT8 variants need no calibrator and no layer pinning — the Q/DQ nodes (only on the
QLAYERS) carry the scales, and TRT runs everything else FP16.

> **Why not one full `images → pose_enc` engine?** It was attempted and does **not build
> on the Orin Nano (7.4 GB)**: the DINO-embed weights inflate the network so TRT can't get
> the ~450 MB of build-time scratch the attention/camera-head region needs (only ~260 MB
> free), and the camera-head fusion (`ForeignNode[...camera_head/Concat]`) can't be placed
> → `Error Code 10`. The blocks engine sidesteps both and still puts the whole transformer
> in TRT. The full-model code paths (`trt_export_pose.py --precision full/full_w8a8`,
> `TRTFullPoseModel`) are kept for boards with more memory (Orin NX/AGX).

### Artifact layout on the SSD

`deploy/artifacts` is a symlink to `/ssd/profiles/quantize` with typed subfolders.
Each ONNX export gets its **own** subdir because `torch.onnx.export` writes external-data
weights as one file per tensor next to the `.onnx` (a shared flat dir would collide):

```
deploy/artifacts/
  ├── onnx/<variant>/<variant>.onnx   # + external-data weight files (kept for inspection)
  ├── engines/<variant>.trt           # + <variant>.trt.json sidecar (arch/TRT/precision/shape)
  ├── timing_cache/orin_sm87.cache    # persistent Orin tactic cache → fast rebuilds
  ├── profiles/                       # trtexec --dumpProfile / *_trtexec.log
  └── results/                        # eval_results_*.json + comparison.md/.csv
```

Paths are centralized in `_trt_bench_common.py` (`onnx_path`/`engine_path`/`results_path`/
`timing_cache_path`); override the root with `$QUANTIZE_ARTIFACTS`. The engine sidecar
lets `trt_inference.py` refuse a foreign engine (TRT engines are **not portable across GPU
architectures** — always build on the target Jetson).

## Approaches → scripts

| # | Approach | Script | Output JSON |
|---|----------|--------|-------------|
| 1 | PyTorch FP16 (baseline) | `eval_co3d.py` | `eval_results.json` |
| 2 | TRT FP32 baseline | `bench_trt_baseline.py --precision fp32` | `eval_results_trt_fp32.json` |
| 3 | TRT FP16 (headline) | `bench_trt_baseline.py --precision fp16` | `eval_results_trt_fp16.json` |
| 4 | TRT INT8 — **LSQ** (headline) | `bench_trt_lsq.py` | `eval_results_trt_w8a8_lsq.json` |
| 5 | TRT INT8 — **ModelOpt PTQ** (headline) | `bench_trt_modelopt.py` | `eval_results_trt_modelopt.json` |
| 6 | TRT INT8 automatic PTQ (calibrator) | `bench_trt_ptq.py` | `eval_results_trt_int8_ptq.json` |
| 7 | PyTorch real-int w8a8 / w8a4 / w4a4 (memory rows) | `eval_co3d_realquant.py --config <cfg>` | `eval_results_<cfg>.json` |
| — | Run the 3 headline engines + compare | `bench_trt_all.py` | (all of 3/4/5) |
| — | Comparison aggregator | `compare_results.py` | `deploy/artifacts/results/comparison.md` + `.csv` |

The **headline trio (3/4/5)** is the FP16 / LSQ / ModelOpt comparison — same blocks
engine, only the QLAYERS' scale source differs. Approaches 5 (LSQ) and 6 (auto-PTQ
calibrator) are two different INT8 *scale sources* on the same layers.

> **Note:** w4a4 / w8a4 stay PyTorch-only memory rows — Orin (SM 8.7) has no INT4
> tensor-core path, so they get no TRT engine. Only **w8a8** maps to a TRT INT8 engine.

## Prerequisites

- Run **inside the Jetson container** (`tensorrt` + `pycuda` available).
- Checkpoint at `data3/rogelio/model_zoo/vggt/vggt_1B_commercial.pt` and the CO3D
  dataset at `data3/rogelio/co3d/...` (override with `--checkpoint` if different).
- Engines/ONNX/results are written under `deploy/artifacts/` (→ `/ssd/profiles/quantize`):
  `onnx/`, `engines/`, `results/`, `profiles/`, `timing_cache/` (see layout above).
- For the `modelopt` variant: `pip install "nvidia-modelopt[torch]"`.

## How to run

### 0. Capability check
```bash
python -c "import tensorrt, pycuda; print(tensorrt.__version__)"
```

### 1. Smoke run (fast — 1 category, 2 sequences)
Run each approach with `--categories apple --max_seqs 2` first to confirm the whole
pipeline works before committing to the full eval:
```bash
python scripts/bench_trt_baseline.py --precision fp16 --categories apple --max_seqs 2
python scripts/bench_trt_baseline.py --precision fp32 --categories apple --max_seqs 2
python scripts/bench_trt_ptq.py                       --categories apple --max_seqs 2
python scripts/bench_trt_lsq.py                       --categories apple --max_seqs 2
python scripts/eval_co3d_realquant.py --config w8a8   --categories apple --max_seqs 2
python scripts/eval_co3d_realquant.py --config w8a4   --categories apple --max_seqs 2
python scripts/eval_co3d_realquant.py --config w4a4   --categories apple --max_seqs 2
python scripts/compare_results.py
```
Each bench script auto-exports the ONNX and builds the engine if missing, then runs the
pose-AUC harness.

### 2. Full run (headline numbers)
Drop `--categories/--max_seqs` to evaluate all 41 CO3D categories:
```bash
python scripts/bench_trt_baseline.py --precision fp16
python scripts/bench_trt_baseline.py --precision fp32
python scripts/bench_trt_ptq.py
python scripts/bench_trt_lsq.py
python scripts/eval_co3d_realquant.py --config w8a8
python scripts/eval_co3d_realquant.py --config w8a4
python scripts/eval_co3d_realquant.py --config w4a4
python scripts/compare_results.py
```

### Manual export / build (optional)
The bench scripts do this automatically, but you can run the stages by hand.

Full pose graph (`images` → `pose_enc`) — the current focus:
```bash
# Full FP16 graph (one fp32 ONNX; engine precision chosen at build)
python scripts/trt_export_pose.py --precision full \
    --out deploy/artifacts/onnx/full_fp16/full_fp16.onnx
# Full graph with LSQ Q/DQ on the QLAYERS
python scripts/trt_export_pose.py --precision full_w8a8 \
    --out deploy/artifacts/onnx/full_int8_lsq/full_int8_lsq.onnx
# Build on THIS Jetson (writes engine + .trt.json sidecar + updates timing cache)
python -m deploy.export.trt_builder \
    --onnx deploy/artifacts/onnx/full_fp16/full_fp16.onnx \
    --output deploy/artifacts/engines/full_fp16.trt --precision fp16
# Or via trtexec (also dumps per-layer profile + uses the timing cache)
bash scripts/trt_build_bench.sh deploy/artifacts/onnx/full_fp16/full_fp16.onnx fp16
```

Blocks graph (`tokens` → `last_tokens`) — the legacy hybrid path:
```bash
python scripts/trt_export_pose.py --precision fp16 \
    --out deploy/artifacts/onnx/pose_baseline/pose_baseline.onnx
python scripts/trt_export_pose.py --precision w8a8 \
    --out deploy/artifacts/onnx/pose_w8a8_lsq/pose_w8a8_lsq.onnx
python scripts/trt_export_pose.py --bench-torch   # PyTorch fp16 pose-only latency
```

## Reading the comparison table

`compare_results.py` prints and writes columns:

| Column | Meaning |
|---|---|
| Mem (MB) | TRT engine file size (TRT rows) or weight memory (PyTorch rows) |
| Latency (ms/seq) | End-to-end per-sequence latency from the harness (includes H2D/D2H for TRT) |
| GPU (ms) | Pure GPU-compute latency via `trtexec --noDataTransfers` (TRT rows only) |
| AUC@30 | Pose accuracy — must stay close to the FP16 baseline for a speedup to count |
| Speedup× | vs the PyTorch FP16 baseline (`eval_results.json`) |
| Mem× | Memory reduction vs the PyTorch FP16 baseline |

### Caveats baked into the numbers
- **Memory metric differs by row** (engine size vs weight memory) — stated in the table
  header. `peak_cuda_mb` from the harness is torch-only and is marked N/A for TRT rows
  (TRT allocates outside PyTorch).
- **TRT inputs are resized to the engine's fixed 350×518.** The resize is identical for
  every TRT approach, so comparisons between TRT rows stay fair; the PyTorch baseline
  uses native preprocessing.
- **"Real acceleration" = the GPU (ms) column.** End-to-end latency includes host
  transfers that a deployed pipeline would overlap/avoid.

## Troubleshooting

- **PTQ quantized the wrong number of layers:** the build log prints
  `Restricted INT8 to N matched compute layers`. Expect `N = 48` (12 blocks × 4
  Linears). If not, TRT fusion renamed layers — adjust the substring match in
  `_restrict_int8_layers` ([deploy/export/trt_builder.py](../deploy/export/trt_builder.py)).
- **No Q/DQ nodes in the LSQ ONNX:** the w8a8 export logs the Q/DQ node count; if it's 0,
  check the `_inject_lsq_scales` → `_replace_qlinear_with_export` handoff in
  [trt_export_pose.py](trt_export_pose.py).
- **Export `Killed` during validation:** that was the old monolithic path OOM-ing when
  `onnx.load` stacked a second multi-GB copy on the torch model. The blocks export now
  frees the torch model first and runs `onnx.checker.check_model(path)` in a subprocess
  (`_validate_onnx`); a checker hiccup is logged as non-fatal since trtexec re-parses.
- **PTQ calibrator shape/`tokens` mismatch:** the engine input is `tokens`, so
  `CO3DCalibrator` needs an `embed_fn` and runs the torch DINO embed on CO3D frames first
  ([bench_trt_ptq.py](bench_trt_ptq.py) builds it). If it logs "no CO3D data found", the
  scales are meaningless — fix the dataset paths in
  [calibration.py](../deploy/export/calibration.py).
- **CUDA context errors:** the TRT runtime uses `pycuda.autoprimaryctx` to share
  PyTorch's primary context; if your PyCUDA is older than 2021.1 it falls back to a
  separate context, which can conflict with the torch-CUDA harness.
- **Calibration warns "no CO3D data found":** check the dataset paths in
  [deploy/export/calibration.py](../deploy/export/calibration.py) (`CO3D_DIR`,
  `CO3D_ANNO_DIR`).
