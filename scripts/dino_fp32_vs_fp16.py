#!/usr/bin/env python3
"""Is torch-fp16 the ground truth, or the anomaly?

Every TRT build of DINO chunk-0 (fp16-strong, fp16-weak, fp32) gives hidden |max|≈74,
while the torch-fp16 golden gives |max|≈125. Before blaming the engine, check torch in
fp32: if torch-fp32 ≈ 74 (matches TRT) then torch-fp16-GPU is amplifying DINOv2's
high-norm outlier tokens and the engines are actually faithful to fp32 — which reframes
the whole gap. If torch-fp32 ≈ 125 (matches torch-fp16) then the engine is genuinely off.

Usage:  python scripts/dino_fp32_vs_fp16.py
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "../src/models/depth/vggt"))

import trt_export_pose as tep


def _stats(name, t):
    a = t.detach().float().cpu().numpy()
    print(f"  {name}: |max|={np.abs(a).max():.2f}  mean|x|={np.abs(a).mean():.4f}")
    return a


def main():
    import eval_co3d_realquant as erq
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    full = erq._load_model(tep.DEFAULT_CKPT, torch.float16, dev)
    full.depth_head = full.point_head = full.track_head = None
    agg = full.aggregator
    B, S = 1, 10
    mid = len(agg.patch_embed.blocks) // 2

    images = torch.zeros(1, 10, 3, tep.DEFAULT_H, tep.DEFAULT_W, dtype=torch.float16, device=dev)

    with torch.inference_mode(), tep._quiet_and_batched():
        sl0 = tep.DinoSlice(agg, B, S, 0, mid, True, False).eval()
        h0_fp16 = sl0(images)
    a16 = _stats("torch h0 (fp16 GPU)", h0_fp16)

    # fp32: cast the ViT chunk to float and rerun on CPU (avoids GPU OOM for the fp32 ViT)
    with torch.inference_mode(), tep._quiet_and_batched():
        agg_cpu = agg.to("cpu", torch.float32)
        sl0_32 = tep.DinoSlice(agg_cpu, B, S, 0, mid, True, False).eval()
        h0_fp32 = sl0_32(images.to("cpu", torch.float32))
    a32 = _stats("torch h0 (fp32 CPU)", h0_fp32)

    d = np.abs(a16 - a32)
    rel = d / (np.abs(a32) + 1e-3)
    print(f"\n  fp16-vs-fp32 torch divergence: max={d.max():.3f} mean={d.mean():.5f} "
          f"rel_mean={rel.mean()*100:.2f}%")
    print("\nIf torch-fp32 |max|≈74 (≈ the TRT engines) → torch-fp16-GPU is the outlier; "
          "the engines are faithful and the 'bug' is fp16 outlier amplification in torch.\n"
          "If torch-fp32 |max|≈125 (≈ torch-fp16) → the engine is genuinely wrong.")


if __name__ == "__main__":
    main()
