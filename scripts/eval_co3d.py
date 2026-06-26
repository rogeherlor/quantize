#!/usr/bin/env python3
"""Standalone CO3D evaluation for the FP16 baseline VGGT model on Jetson.

Usage:
    python scripts/eval_co3d.py
    python scripts/eval_co3d.py --checkpoint data3/rogelio/model_zoo/vggt/vggt_1B_commercial.pt
"""
import os
import sys

# Disable expandable_segments: Jetson NvMap does not support VMM segment
# expansion. With expandable_segments enabled (PyTorch 2.7+ default), the
# caching allocator tries to extend existing NvMap segments during inference
# rather than sub-allocating from the pre-established pool. On Jetson this
# triggers NvMap ENOMEM on every activation allocation → NVML_SUCCESS assert.
os.environ.pop("PYTORCH_NO_CUDA_MEMORY_CACHING", None)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

# ── Drop page cache and compact physical memory before CUDA initialises ───────
# model.to('cuda') makes 1797 individual NvMap calls. With a fragmented page
# allocator each call can fail with ENOMEM. Clearing the cache and compacting
# first gives NvMap large contiguous regions to work with.
try:
    with open("/proc/sys/vm/drop_caches", "w") as _f:
        _f.write("3\n")
except OSError:
    pass
try:
    with open("/proc/sys/vm/compact_memory", "w") as _f:
        _f.write("1\n")
    import time as _t; _t.sleep(3); del _t
except OSError:
    pass

import argparse
import gc
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/models/depth/vggt"))

from src.logger import logger
from vggt.models.vggt import VGGT
from src.models.depth.qvggt import run_evaluation_vggt

DEFAULT_CKPT = "data3/rogelio/model_zoo/vggt/vggt_1B_commercial.pt"
# Activation memory scales ~130 MB/frame (Flash Attention, empirical).
# 10 frames × 130 + 300 MB buffer = 1600 MB.
ACT_POOL_MB = 1600


def _alloc_pool(target_mb, device, min_chunk_mb=32):
    chunks, pooled = [], 0
    chunk = 512
    while pooled < target_mb and chunk >= min_chunk_mb:
        want = min(chunk, target_mb - pooled)
        try:
            chunks.append(torch.empty(want * 1024 * 1024 // 4, dtype=torch.float32, device=device))
            pooled += want
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            chunk //= 2
    return chunks, pooled


def _load_model(checkpoint_path: str, dtype: torch.dtype, device: str) -> nn.Module:
    """Two-phase pool + meta device + mmap streaming — same strategy as infer.py.

    Phase 1 (model pool, 2700 MB): kept alive while phase 2 runs so phase 2
    is forced into FRESH NvMap segments (free list is empty).
    Phase 2 (activation pool, 1600 MB): fresh segments, never fragmented by
    model params because they haven't been allocated yet.
    Model pool freed → model loads from those segments (fragments them).
    Activation pool freed AFTER model load → 1600 MB contiguous free for inference.
    No NvMap calls needed during the forward pass.
    """
    # Second compact_memory: PyTorch CUDA init re-fragments pages.
    try:
        with open("/proc/sys/vm/compact_memory", "w") as _f:
            _f.write("1\n")
        time.sleep(3)
    except OSError:
        pass

    cuda_free_mb = torch.cuda.mem_get_info()[0] // (1024 ** 2)
    model_pool_mb = min(2700, cuda_free_mb - ACT_POOL_MB - 200)

    logger.info(f"Pre-allocating model pool: {model_pool_mb} MB (kept alive) ...")
    _model_chunks, _model_pooled = _alloc_pool(model_pool_mb, device)

    logger.info(f"Pre-allocating activation pool: {ACT_POOL_MB} MB "
                f"(separate NvMap segments — model pool still alive) ...")
    _act_chunks, _act_pooled = _alloc_pool(ACT_POOL_MB, device)

    # Release model pool → model will load into these segments.
    del _model_chunks
    gc.collect()
    logger.info(f"Pools ready: model {_model_pooled} MB + activation {_act_pooled} MB  "
                f"| CUDA free: {torch.cuda.mem_get_info()[0] // (1024**2)} MB")

    # Build model on meta device (zero physical RAM).
    _orig_item = torch.Tensor.item
    torch.Tensor.item = lambda self: 0 if self.is_meta else _orig_item(self)
    try:
        with torch.device("meta"):
            model = VGGT().eval()
    finally:
        torch.Tensor.item = _orig_item

    # Stream checkpoint from NVMe mmap into CUDA pool.
    # mmap=True: tensors are demand-paged from NVMe, never fully in CPU RAM.
    logger.info(f"Streaming checkpoint ({dtype}) from {checkpoint_path} ...")
    state_dict = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    model_keys = (set(k for k, _ in model.named_parameters()) |
                  set(k for k, _ in model.named_buffers()))
    cuda_sd = {}
    for key in list(state_dict.keys()):
        t = state_dict[key]
        if isinstance(t, torch.Tensor) and key in model_keys:
            cuda_sd[key] = t.to(device, dtype=dtype)
        del state_dict[key]
    del state_dict
    gc.collect()

    missing, unexpected = model.load_state_dict(cuda_sd, assign=True, strict=False)
    del cuda_sd
    gc.collect()
    logger.info(f"Loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")

    # Drop the mmap page cache (~4.8 GB of FP32 pages left by checkpoint streaming).
    try:
        with open("/proc/sys/vm/drop_caches", "w") as _f:
            _f.write("3\n")
        with open("/proc/sys/vm/compact_memory", "w") as _f:
            _f.write("1\n")
        time.sleep(2)
    except OSError:
        pass

    # Release activation pool → caching allocator free list.
    # These blocks came from separate NvMap segments so they are fully contiguous.
    del _act_chunks
    gc.collect()
    logger.info(f"Activation pool released: {_act_pooled} MB contiguous free for inference  "
                f"| CUDA allocated: {torch.cuda.memory_allocated() // (1024**2)} MB")

    # Fix non-persistent ImageNet normalisation buffers not stored in checkpoint.
    for module in model.modules():
        for name, val in (("_resnet_mean", [0.485, 0.456, 0.406]),
                          ("_resnet_std",  [0.229, 0.224, 0.225])):
            buf = getattr(module, name, None)
            if buf is not None and (buf.is_meta or buf.device.type != device):
                module.register_buffer(
                    name,
                    torch.tensor(val, dtype=dtype, device=device).view(1, 1, 3, 1, 1),
                    persistent=False,
                )

    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--categories", default=None,
                   help="comma-separated category list for a fast estimate, e.g. 'apple'.")
    p.add_argument("--max_seqs", type=int, default=None,
                   help="cap sequences per category (e.g. 5) for a quick run.")
    p.add_argument("--results", default=None, help="output JSON path")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    logger.info(f"Device: {device}  |  SM{torch.cuda.get_device_capability()[0]}.x  |  dtype: {dtype}")

    model = _load_model(args.checkpoint, dtype, device)
    categories = [c.strip() for c in args.categories.split(",")] if args.categories else None
    run_evaluation_vggt(model, categories=categories, max_seqs=args.max_seqs,
                        results_path=args.results)


if __name__ == "__main__":
    main()
