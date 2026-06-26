#!/usr/bin/env python3
"""Isolate the resolution contribution to the TRT accuracy gap.

The TRT/INT8 chains are fixed-shape and resize every frame to 350×518, while the PyTorch
baseline (0.814) uses each frame's NATIVE load_and_preprocess size (width 518, height
350–518 per aspect). This runs the PyTorch pose model but FORCES the same 350×518 resize the
TRT path applies (deploy/runtime/trt_inference.py), so:

    0.814  − AUC_torch@350x518  = the RESOLUTION loss
    AUC_torch@350x518 − 0.653   = the TRT-fp16 execution loss

Lighter than the native baseline (fewer tokens), so it's GPU-safe.

Usage:  python scripts/eval_torch_res.py --height 350 --width 518 --categories apple --max_seqs 2
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "../src/models/depth/vggt"))

import trt_export_pose as tep
import eval_co3d_realquant as erq
from src.logger import logger
from src.models.depth.qvggt import run_evaluation_vggt


class _ResizeWrap(nn.Module):
    """Resize every frame to (H,W) before the model — exactly the resize TRTChainedPoseModel
    does — so a pure-torch run matches the TRT path's input resolution."""
    def __init__(self, model, hw):
        super().__init__()
        self.model = model
        self.hw = hw

    def forward(self, images, **kwargs):
        *lead, C, H, W = images.shape
        if (H, W) != self.hw:
            flat = images.reshape(-1, C, H, W)
            flat = F.interpolate(flat, size=self.hw, mode="bilinear", align_corners=False)
            images = flat.reshape(*lead, C, *self.hw)
        return self.model(images, **kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=tep.DEFAULT_CKPT)
    p.add_argument("--height", type=int, default=350)
    p.add_argument("--width", type=int, default=518)
    p.add_argument("--categories", default=None)
    p.add_argument("--max_seqs", type=int, default=None)
    p.add_argument("--results", default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading full VGGT (fp16) for torch eval @ {args.height}x{args.width} ...")
    full = erq._load_model(args.checkpoint, torch.float16, device)
    full.depth_head = full.point_head = full.track_head = None   # pose-only (matches baseline)
    model = _ResizeWrap(full, (args.height, args.width)).eval()

    cats = [c.strip() for c in args.categories.split(",")] if args.categories else None
    results = args.results or os.path.join(
        "deploy/artifacts/results", f"eval_results_torch_{args.height}x{args.width}.json")
    os.makedirs(os.path.dirname(results) or ".", exist_ok=True)
    logger.info(f"Evaluating PyTorch fp16 @ {args.height}x{args.width} → {results}")
    run_evaluation_vggt(model, model_path=f"PyTorch fp16 @ {args.height}x{args.width}",
                        results_path=results, categories=cats, max_seqs=args.max_seqs)


if __name__ == "__main__":
    main()
