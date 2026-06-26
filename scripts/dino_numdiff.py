#!/usr/bin/env python3
"""Decisive localization of the all-TRT collapse: are the DINO engines faithful?

By elimination (trt_numdiff.py: transformer chain ≈ 2.2%; camhead_numdiff.py: camera
head ≈ 2.2% at matched resolution), the only remaining difference between the hybrid
path (AUC 0.65) and the full all-TRT path (AUC 0.17) is the DINO embed split
(dino_0 → dino_1). This compares the TRT DINO chain output against torch
`Aggregator.embed_tokens` on the SAME 350×518 images.

  torch:  images[1,S,3,350,518] → embed_tokens → tokens [B*S, P, C]  (the hybrid's path)
  TRT:    images → dino_0 → dino_1 → tokens

A large error here = the DINO engines are the dominant cause of the all-TRT collapse.

Usage (inside the Jetson container):
  python scripts/dino_numdiff.py            # zeros input (no dataset needed)
  python scripts/dino_numdiff.py --real     # one real CO3D sequence, resized to 350×518
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
    a = a.astype(np.float32); b = b.astype(np.float32)
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
    p.add_argument("--real", action="store_true", help="use one real CO3D sequence")
    args = p.parse_args()

    import eval_co3d_realquant as erq
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    full = erq._load_model(tep.DEFAULT_CKPT, torch.float16, dev)
    full.depth_head = full.point_head = full.track_head = None
    agg = full.aggregator

    if args.real:
        import gzip, json
        from vggt.utils.load_fn import load_and_preprocess_images
        images = None
        for category in erq.CALIB_CATEGORIES:
            anno = os.path.join(erq.CO3D_ANNO_DIR, f"{category}_test.jgz")
            if not os.path.exists(anno):
                continue
            with gzip.open(anno, "r") as fin:
                annotation = json.loads(fin.read())
            for seq_name in sorted(annotation.keys()):
                sd = annotation[seq_name]
                if len(sd) < args.frames:
                    continue
                ids = np.random.choice(len(sd), args.frames, replace=False)
                names = [os.path.join(erq.CO3D_DIR, sd[i]["filepath"]) for i in ids]
                imgs = load_and_preprocess_images(names).to(dev, torch.float16)
                if tuple(imgs.shape[-2:]) != (tep.DEFAULT_H, tep.DEFAULT_W):
                    imgs = torch.nn.functional.interpolate(
                        imgs, size=(tep.DEFAULT_H, tep.DEFAULT_W),
                        mode="bilinear", align_corners=False)
                images = imgs.unsqueeze(0) if imgs.dim() == 4 else imgs
                break
            if images is not None:
                break
        if images is None:
            raise RuntimeError("No CO3D sequence found.")
    else:
        images = torch.zeros(1, args.frames, 3, tep.DEFAULT_H, tep.DEFAULT_W,
                             dtype=torch.float16, device=dev)

    with torch.inference_mode(), tep._quiet_and_batched():
        tokens, pos, B, S, P, C = agg.embed_tokens(images)
    tok_torch = tokens.detach().float().cpu().numpy()
    img_np = images.detach().float().cpu().numpy()
    print(f"\ntorch tokens {tuple(tok_torch.shape)}  |max|={np.abs(tok_torch).max():.1f}")

    # Free the model, defrag, load the two DINO engines.
    full.aggregator = None
    del agg, full, tokens, images
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
        try:
            blk = torch.empty(800 * 1024 * 1024 // 2, dtype=torch.float16, device=dev)
            del blk
        except Exception:
            pass
        torch.cuda.empty_cache()

    from deploy.runtime.trt_inference import TRTInferenceEngine
    e0 = TRTInferenceEngine(args.dino0)
    e1 = TRTInferenceEngine(args.dino1)
    hidden = e0.infer({_in_name(e0): img_np})[e0._output_names[0]]
    tok_trt = e1.infer({_in_name(e1): hidden})[e1._output_names[0]]
    e0.close(); e1.close()

    print(f"TRT tokens {tuple(np.asarray(tok_trt).shape)}  |max|={np.abs(tok_trt).max():.1f}")
    print("\n=== DINO (dino_0→dino_1) TRT vs torch embed_tokens ===")
    print(" ", _err(tok_trt, tok_torch))
    print("\nLarge error → the DINO engines are the dominant cause of the all-TRT collapse "
          "(structural/precision issue in the ViT split or token assembly). Small error → "
          "the collapse is compounding across all three stages, not one bad engine.")


if __name__ == "__main__":
    main()
