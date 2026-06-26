"""Shared helpers for the per-approach TensorRT benchmark scripts.

Every approach (bench_trt_baseline / bench_trt_ptq / bench_trt_lsq) wraps its TRT
engine in TRTPoseModel and runs the SAME metric harness (run_evaluation_vggt), so
the resulting JSON is directly comparable. These helpers cover the common steps:
ONNX export-if-missing, engine eval, and JSON augmentation (engine size + pure-GPU
trtexec latency), since the harness's torch-only peak_cuda is meaningless for TRT.
"""
import argparse
import json
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
for _p in (_ROOT, os.path.join(_ROOT, "src/models/depth/vggt")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_CKPT = "data3/rogelio/model_zoo/vggt/vggt_1B_commercial.pt"
DEFAULT_FRAMES, DEFAULT_H, DEFAULT_W = 10, 350, 518

# ── Artifact layout (deploy/artifacts → /ssd/profiles/quantize) ───────────────
# Big artifacts live on the SSD via the in-repo `deploy/artifacts` symlink. Each
# ONNX export gets its OWN subdir because torch.onnx.export writes external-data
# weights as one file per tensor (named by tensor) next to the .onnx — a shared
# flat dir would collide across exports. Engines are single self-contained files,
# so they share one flat engines/ dir. Override the root with $QUANTIZE_ARTIFACTS.
ARTIFACT_ROOT = os.environ.get(
    "QUANTIZE_ARTIFACTS", os.path.join(_ROOT, "deploy", "artifacts"))


def _ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def onnx_path(variant):
    """<root>/onnx/<variant>/<variant>.onnx — own dir so external data can't collide."""
    return os.path.join(_ensure(os.path.join(ARTIFACT_ROOT, "onnx", variant)),
                        variant + ".onnx")


def engine_path(variant):
    """<root>/engines/<variant>.trt"""
    return os.path.join(_ensure(os.path.join(ARTIFACT_ROOT, "engines")),
                        variant + ".trt")


def results_path(variant):
    """<root>/results/eval_results_<variant>.json"""
    return os.path.join(_ensure(os.path.join(ARTIFACT_ROOT, "results")),
                        "eval_results_" + variant + ".json")


def profiles_dir():
    return _ensure(os.path.join(ARTIFACT_ROOT, "profiles"))


def timing_cache_path():
    """Persistent TRT timing cache — Jetson(Orin SM 8.7)-specific, speeds rebuilds."""
    return os.path.join(_ensure(os.path.join(ARTIFACT_ROOT, "timing_cache")),
                        "orin_sm87.cache")


def export_args(out, precision, checkpoint=DEFAULT_CKPT, frames=DEFAULT_FRAMES,
                height=DEFAULT_H, width=DEFAULT_W, calib_seqs=2, calib_frames=4):
    """Build the Namespace expected by scripts/trt_export_pose.py exporters."""
    return argparse.Namespace(
        out=out, precision=precision, checkpoint=checkpoint, frames=frames,
        height=height, width=width, calib_seqs=calib_seqs, calib_frames=calib_frames)


def ensure_baseline_onnx(onnx_path, **kw):
    if not os.path.exists(onnx_path):
        import trt_export_pose as tep
        tep.export_baseline(export_args(onnx_path, "fp32", **kw))
    return onnx_path


def ensure_w8a8_onnx(onnx_path, **kw):
    if not os.path.exists(onnx_path):
        import trt_export_pose as tep
        tep.export_w8a8(export_args(onnx_path, "w8a8", **kw))
    return onnx_path


def ensure_modelopt_onnx(onnx_path, checkpoint=DEFAULT_CKPT, calib_seqs=2, calib_frames=4,
                         algorithm="max", weight_per_channel=True):
    """Blocks ONNX (tokens → last_tokens) with ModelOpt INT8 Q/DQ on the QLAYERS —
    the standardized-tool counterpart to ensure_w8a8_onnx (learned LSQ). `algorithm` +
    `weight_per_channel` select the calibration config (must match the fake-quant eval)."""
    if not os.path.exists(onnx_path):
        import argparse
        import export_modelopt_ptq as mo
        mo.export_modelopt(argparse.Namespace(
            checkpoint=checkpoint, out=onnx_path, frames=DEFAULT_FRAMES,
            height=DEFAULT_H, width=DEFAULT_W,
            calib_seqs=calib_seqs, calib_frames=calib_frames,
            algorithm=algorithm, weight_per_channel=weight_per_channel))
    return onnx_path


def ensure_full_onnx(onnx_path, **kw):
    """Full pose graph (images → pose_enc), un-quantized. One fp32 ONNX serves the
    full_fp16 engine (precision chosen at build)."""
    if not os.path.exists(onnx_path):
        import trt_export_pose as tep
        tep.export_full(export_args(onnx_path, "full", **kw))
    return onnx_path


def ensure_full_w8a8_onnx(onnx_path, **kw):
    """Full pose graph with Q/DQ on the QLAYERS (images → pose_enc) for the
    full_int8_lsq engine."""
    if not os.path.exists(onnx_path):
        import trt_export_pose as tep
        tep.export_full_w8a8(export_args(onnx_path, "full_w8a8", **kw))
    return onnx_path


def eval_all_engines(engine_paths, results_path, label, categories=None, max_seqs=None,
                     height=DEFAULT_H, width=DEFAULT_W, num_frames=DEFAULT_FRAMES):
    """Evaluate config #3 — the WHOLE pose path in TRT (no torch compute) as a chain of
    single-I/O engines: dino_0 → dino_1 → chain_s0 → chain_s8 → chain_s16 → camera_head.
    engine_paths must be in execution order (dino_0 first, camera_head last)."""
    from deploy.runtime.trt_inference import TRTAllEngineModel
    from src.models.depth.qvggt import run_evaluation_vggt
    from src.logger import logger

    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    cats = [c.strip() for c in categories.split(",")] if isinstance(categories, str) else categories
    logger.info(f"[{label}] evaluating {len(engine_paths)}-engine all-TRT path → {results_path}")
    model = TRTAllEngineModel(engine_paths, height=height, width=width)
    try:
        run_evaluation_vggt(model, model_path=label, results_path=results_path,
                            categories=cats, max_seqs=max_seqs, num_frames=num_frames)
    finally:
        model.close()
    total_mb = round(sum(os.path.getsize(p) for p in engine_paths) / 1024 / 1024, 1)
    with open(results_path) as f:
        res = json.load(f)
    res["engine_mb"] = total_mb
    res["peak_cuda_mb"] = None
    res["runtime"] = "tensorrt-all"
    res["n_engines"] = len(engine_paths)
    with open(results_path, "w") as f:
        json.dump(res, f, indent=2)
    logger.info(f"[{label}] AUC@30={res.get('mean_auc', {}).get('auc30')}  "
                f"engines={len(engine_paths)} total={total_mb}MB")
    return res


def trtexec_gpu_ms(engine_path):
    """Pure GPU-compute latency (ms, median) via trtexec --noDataTransfers.
    Returns None if trtexec is unavailable or parsing fails."""
    trtexec = None
    for cand in ("trtexec", "/usr/src/tensorrt/bin/trtexec"):
        if subprocess.run(["bash", "-lc", f"command -v {cand}"],
                          capture_output=True).returncode == 0 or os.path.exists(cand):
            trtexec = cand
            break
    if trtexec is None:
        return None
    try:
        out = subprocess.run(
            [trtexec, f"--loadEngine={engine_path}", "--noDataTransfers",
             "--iterations=50", "--avgRuns=20", "--warmUp=2000"],
            capture_output=True, text=True, timeout=600).stdout
    except Exception:
        return None
    m = re.search(r"GPU Compute Time:.*?median\s*=\s*([0-9.]+)\s*ms", out)
    return float(m.group(1)) if m else None


def augment_results(results_path, engine_path):
    """Add engine file size + trtexec GPU latency to the harness JSON; mark the
    torch-only peak_cuda_mb as N/A for TRT engines."""
    with open(results_path) as f:
        res = json.load(f)
    res["engine_mb"] = round(os.path.getsize(engine_path) / 1024 / 1024, 1)
    res["trtexec_gpu_ms"] = trtexec_gpu_ms(engine_path)
    res["peak_cuda_mb"] = None  # torch peak does not capture TRT allocations
    res["runtime"] = "tensorrt"
    with open(results_path, "w") as f:
        json.dump(res, f, indent=2)
    return res


def eval_engine(engine_path, results_path, label, categories=None, max_seqs=None,
                checkpoint=DEFAULT_CKPT, height=DEFAULT_H, width=DEFAULT_W, full=False,
                num_frames=DEFAULT_FRAMES):
    """Run the standard pose-AUC harness over a TRT engine and write a comparable JSON.

    full=False → hybrid TRTPoseModel (torch DINO embed + TRT blocks + torch camera head).
    full=True  → TRTFullPoseModel (images → pose_enc entirely in TRT; no torch parts)."""
    from src.models.depth.qvggt import run_evaluation_vggt
    from src.logger import logger

    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    cats = [c.strip() for c in categories.split(",")] if isinstance(categories, str) else categories
    kind = "full" if full else "blocks"
    logger.info(f"[{label}] evaluating TRT {kind} engine {engine_path} → {results_path}")
    if full:
        from deploy.runtime.trt_inference import TRTFullPoseModel
        model = TRTFullPoseModel(engine_path, height=height, width=width)
    else:
        from deploy.runtime.trt_inference import TRTPoseModel
        model = TRTPoseModel(engine_path, checkpoint=checkpoint, height=height, width=width)
    try:
        run_evaluation_vggt(model, model_path=label, results_path=results_path,
                            categories=cats, max_seqs=max_seqs, num_frames=num_frames)
    finally:
        model.close()
    res = augment_results(results_path, engine_path)
    logger.info(f"[{label}] AUC@30={res.get('mean_auc', {}).get('auc30')}  "
                f"engine={res['engine_mb']}MB  gpu={res['trtexec_gpu_ms']}ms")
    return res


def eval_chain(engine_paths, results_path, label, categories=None, max_seqs=None,
               checkpoint=DEFAULT_CKPT, height=DEFAULT_H, width=DEFAULT_W,
               num_frames=DEFAULT_FRAMES):
    """Evaluate a CHAIN of TRT sub-engines (transformer split across N engines) with
    the shared pose-AUC harness. engine_paths in pair order (chunk 0 first)."""
    from deploy.runtime.trt_inference import TRTChainedPoseModel
    from src.models.depth.qvggt import run_evaluation_vggt
    from src.logger import logger

    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    cats = [c.strip() for c in categories.split(",")] if isinstance(categories, str) else categories
    logger.info(f"[{label}] evaluating {len(engine_paths)}-engine chain → {results_path}")
    model = TRTChainedPoseModel(engine_paths, checkpoint=checkpoint, height=height, width=width)
    try:
        run_evaluation_vggt(model, model_path=label, results_path=results_path,
                            categories=cats, max_seqs=max_seqs, num_frames=num_frames)
    finally:
        model.close()
    # Sum the chunk engine sizes; trtexec GPU time isn't meaningful for a chain.
    total_mb = round(sum(os.path.getsize(p) for p in engine_paths) / 1024 / 1024, 1)
    with open(results_path) as f:
        res = json.load(f)
    res["engine_mb"] = total_mb
    res["peak_cuda_mb"] = None
    res["runtime"] = "tensorrt-chain"
    res["n_engines"] = len(engine_paths)
    with open(results_path, "w") as f:
        json.dump(res, f, indent=2)
    logger.info(f"[{label}] AUC@30={res.get('mean_auc', {}).get('auc30')}  "
                f"engines={len(engine_paths)} total={total_mb}MB")
    return res
