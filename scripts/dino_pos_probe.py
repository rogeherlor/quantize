#!/usr/bin/env python3
"""Confirm (in pure torch, no engine) that the DINO pos-encoding is what corrupts tokens.

DINOv2 computes interpolate_pos_encoding in fp32 with bicubic + antialias + an offset
scale-factor kludge. The ONNX(opset17)→fp16→TRT path instead runs a fp16 cubic Resize
with a fixed output size (no antialias, no offset). This isolates ONLY that difference:
it runs the SAME patch-embed and the SAME first ViT blocks twice — once with the real
pos-encoding, once with an "engine-like" pos-encoding — and measures the token error.

If the error is ~hundreds of %, the pos-encoding is the cause of the all-TRT collapse
and the fix is to bake the fp32 pos-encoding as a constant (the export is fixed-shape).

Usage:  python scripts/dino_pos_probe.py
"""
import math
import os
import sys
import types

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "../src/models/depth/vggt"))

import trt_export_pose as tep


def _err(a, b):
    a = a.detach().float().cpu().numpy(); b = b.detach().float().cpu().numpy()
    d = np.abs(a - b)
    rel = d / (np.abs(b) + 1e-3)
    return f"max={d.max():.4f}  mean={d.mean():.5f}  rel_mean={rel.mean()*100:.2f}%"


def _engine_like_pos(vit):
    """Reproduce the exported ONNX/TRT pos-encoding: same dtype as the running tensor
    (fp16 in the engine), bicubic, antialias=False, fixed output size, no offset."""
    def f(self, x, w, h):
        N = self.pos_embed.shape[1] - 1
        M = int(math.sqrt(N))
        dim = x.shape[-1]
        w0, h0 = w // self.patch_size, h // self.patch_size
        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        pp = patch_pos.reshape(1, M, M, dim).permute(0, 3, 1, 2)   # keep current dtype (fp16)
        pp = torch.nn.functional.interpolate(pp, size=(w0, h0), mode="bicubic", antialias=False)
        pp = pp.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((cls_pos, pp), dim=1).to(x.dtype)
    return types.MethodType(f, vit)


def main():
    import eval_co3d_realquant as erq
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    full = erq._load_model(tep.DEFAULT_CKPT, torch.float16, dev)
    full.depth_head = full.point_head = full.track_head = None
    vit = full.aggregator.patch_embed

    images = torch.zeros(2, 3, tep.DEFAULT_H, tep.DEFAULT_W, dtype=torch.float16, device=dev)

    orig = vit.interpolate_pos_encoding
    n_blocks = 2
    # the [cls + patch] tensor that interpolate_pos_encoding is called on inside
    # prepare_tokens_with_masks (before register tokens are inserted)
    with torch.inference_mode(), tep._quiet_and_batched():
        cls_plus_patch = torch.cat(
            (vit.cls_token.expand(images.shape[0], -1, -1), vit.patch_embed(images)), dim=1)
        pos_ref = orig(cls_plus_patch, tep.DEFAULT_H, tep.DEFAULT_W)

        # real (fp32 bicubic + antialias + offset) pos-encoding path
        x_ref = vit.prepare_tokens_with_masks(images)
        for blk in vit.blocks[:n_blocks]:
            x_ref = blk(x_ref)

        # engine-like (fp16 cubic, no antialias, fixed size) pos-encoding path
        vit.interpolate_pos_encoding = _engine_like_pos(vit)
        pos_alt = vit.interpolate_pos_encoding(cls_plus_patch, tep.DEFAULT_H, tep.DEFAULT_W)
        x_alt = vit.prepare_tokens_with_masks(images)
        for blk in vit.blocks[:n_blocks]:
            x_alt = blk(x_alt)

    print("\n=== pos-encoding tensor: real(fp32 bicubic+aa+offset) vs engine-like(fp16 cubic) ===")
    print(" ", _err(pos_alt, pos_ref))
    print(f"=== ViT hidden after {n_blocks} blocks: engine-like pos vs real pos ===")
    print(" ", _err(x_alt, x_ref))
    print("\nLarge error → the pos-encoding is the cause; fix = bake fp32 pos as a constant.")


if __name__ == "__main__":
    main()
