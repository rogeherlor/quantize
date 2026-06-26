#!/usr/bin/env python3
"""Localize WHERE the DINO TRT path diverges (the ~270% all-TRT collapse).

Three comparisons in one model load:
  A. torch DinoSlice composition (chunk0∘chunk1)  vs  torch embed_tokens
     → pure-torch EXPORT-LOGIC check. Large = the DinoSlice wrapper is wrong (stage A).
  B. dino_0.trt output  vs  torch DinoSlice chunk0 hidden
     → does chunk-0 (prepare_tokens + first 12 ViT blocks) survive ONNX/fp16/TRT?
  C. dino_1.trt(torch chunk0)  vs  torch DinoSlice chunk1 tokens
     → does chunk-1 (last 12 blocks + norm + token assembly) survive?

Whichever of A/B/C explodes is the culprit stage.

Usage:  python scripts/dino_stage_probe.py [--real]
"""
import argparse
import gc
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "../src/models/depth/vggt"))

import trt_export_pose as tep


def _err(a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    d = np.abs(a - b)
    rel = d / (np.abs(b) + 1e-3)
    return f"max={d.max():.4f}  mean={d.mean():.5f}  rel_mean={rel.mean()*100:.2f}%"


def _in_name(eng):
    return [eng.engine.get_tensor_name(i) for i in range(eng.engine.num_io_tensors)
            if eng.engine.get_tensor_mode(eng.engine.get_tensor_name(i))
            == eng._trt.TensorIOMode.INPUT][0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dino0", default="deploy/artifacts/engines/dino_0.trt")
    p.add_argument("--dino1", default="deploy/artifacts/engines/dino_1.trt")
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--real", action="store_true")
    args = p.parse_args()

    import eval_co3d_realquant as erq
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    full = erq._load_model(tep.DEFAULT_CKPT, torch.float16, dev)
    full.depth_head = full.point_head = full.track_head = None
    agg = full.aggregator
    B, S = 1, args.frames

    if args.real:
        import gzip, json
        from vggt.utils.load_fn import load_and_preprocess_images
        images = None
        for cat in erq.CALIB_CATEGORIES:
            anno = os.path.join(erq.CO3D_ANNO_DIR, f"{cat}_test.jgz")
            if not os.path.exists(anno):
                continue
            with gzip.open(anno, "r") as fin:
                ann = json.loads(fin.read())
            for sq in sorted(ann.keys()):
                sd = ann[sq]
                if len(sd) < args.frames:
                    continue
                ids = np.random.choice(len(sd), args.frames, replace=False)
                names = [os.path.join(erq.CO3D_DIR, sd[i]["filepath"]) for i in ids]
                im = load_and_preprocess_images(names).to(dev, torch.float16)
                if tuple(im.shape[-2:]) != (tep.DEFAULT_H, tep.DEFAULT_W):
                    im = torch.nn.functional.interpolate(
                        im, size=(tep.DEFAULT_H, tep.DEFAULT_W), mode="bilinear", align_corners=False)
                images = im.unsqueeze(0) if im.dim() == 4 else im
                break
            if images is not None:
                break
    else:
        images = torch.zeros(1, args.frames, 3, tep.DEFAULT_H, tep.DEFAULT_W,
                             dtype=torch.float16, device=dev)

    depth = len(agg.patch_embed.blocks)
    mid = depth // 2
    with torch.inference_mode(), tep._quiet_and_batched():
        tok_embed = agg.embed_tokens(images)[0]                       # production reference
        sl0 = tep.DinoSlice(agg, B, S, 0, mid, True, False).eval()
        h0 = sl0(images)                                             # torch chunk-0 hidden
        sl1 = tep.DinoSlice(agg, B, S, mid, depth, False, True).eval()
        tok_slice = sl1(h0)                                         # torch DinoSlice composition

    tok_embed_np = tok_embed.detach().float().cpu().numpy()
    h0_np = h0.detach().float().cpu().numpy()
    tok_slice_np = tok_slice.detach().float().cpu().numpy()
    img_np = images.detach().float().cpu().numpy()   # exact same input for the engine
    print(f"\nimages {tuple(images.shape)}  embed_tokens {tok_embed_np.shape}  "
          f"chunk0 hidden {h0_np.shape}")
    print("\n=== A. torch DinoSlice(chunk0∘chunk1) vs torch embed_tokens (export logic) ===")
    print(" ", _err(tok_slice_np, tok_embed_np))

    del sl0, sl1, tok_embed, h0, tok_slice, agg, full, images
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
        try:
            blk = torch.empty(800 * 1024 * 1024 // 2, dtype=torch.float16, device=dev); del blk
        except Exception:
            pass
        torch.cuda.empty_cache()

    def _rng(name, a):
        a = np.asarray(a, np.float32)
        fin = np.isfinite(a)
        big = int((np.abs(a) > 256).sum())   # |x|>256 ⇒ x² overflows fp16 (max 65504)
        print(f"  range[{name}]: |max|={np.abs(a[fin]).max():.1f}  inf/nan={int((~fin).sum())}  "
              f"#|x|>256={big}")

    print("\n=== magnitudes of the chunk-0 hidden residual stream ===")
    _rng("torch h0", h0_np)

    from deploy.runtime.trt_inference import TRTInferenceEngine
    e0 = TRTInferenceEngine(args.dino0)
    e1 = TRTInferenceEngine(args.dino1)
    # re-feed the SAME images we used in torch:
    h0_trt = e0.infer({_in_name(e0): img_np})[e0._output_names[0]]
    _rng("TRT h0", h0_trt)
    print("\n=== B. dino_0.trt vs torch chunk-0 hidden (chunk-0 engine) ===")
    print(" ", _err(h0_trt, h0_np))
    tok_trt = e1.infer({_in_name(e1): h0_np.astype(np.float16)})[e1._output_names[0]]
    print("=== C. dino_1.trt(torch chunk0) vs torch chunk1 tokens (chunk-1 engine) ===")
    print(" ", _err(tok_trt, tok_slice_np))
    e0.close(); e1.close()


if __name__ == "__main__":
    main()
