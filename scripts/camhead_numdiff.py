#!/usr/bin/env python3
"""Localize the all-TRT collapse: is the CAMERA HEAD the culprit?

The hybrid path (transformer in TRT, DINO embed + camera head in PyTorch) keeps
AUC@30 ≈ 0.65, but the full all-TRT path (DINO + transformer + camera head all in TRT)
collapses to ≈ 0.17. trt_numdiff.py already showed the transformer chain only adds
~2.2% token error, so the collapse must come from the dino_*/camera_head engines.

This isolates the CAMERA HEAD: it runs ONE real CO3D sequence through the full torch
fp16 model to obtain (a) the real camera-head input `last_tokens` [1,S,P,2C] and
(b) the reference torch `pose_enc` [1,S,9], then runs the SAME last_tokens through the
camera_head.trt engine and reports the pose_enc error. It also logs the fp16 dynamic
range (max-abs / inf / NaN) of last_tokens and pose_enc to test the overflow hypothesis.

Memory choreography mirrors trt_numdiff.py: the torch model is dropped and NvMap
defragged before the engine loads, so the 2.4 GB model and the 0.4 GB engine never
co-reside on the 7.8 GB board.

Usage (inside the Jetson container):
  python scripts/camhead_numdiff.py
  python scripts/camhead_numdiff.py --engine deploy/artifacts/engines/camera_head.trt
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


def _range(name, t):
    a = t.detach().float().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t, np.float32)
    finite = np.isfinite(a)
    n_bad = int((~finite).sum())
    amax = float(np.abs(a[finite]).max()) if finite.any() else float("nan")
    flag = "  <-- exceeds fp16 max (65504)!" if amax > 60000 else ""
    print(f"  range[{name}]: |max|={amax:.1f}  inf/nan={n_bad}  shape={tuple(a.shape)}{flag}")


def _one_real_sequence(num_frames, device, dtype):
    """Load ONE real CO3D sequence [1,num_frames,3,H,W] (same source as calibration)."""
    import gzip, json
    import eval_co3d_realquant as erq
    from vggt.utils.load_fn import load_and_preprocess_images
    for category in erq.CALIB_CATEGORIES:
        anno = os.path.join(erq.CO3D_ANNO_DIR, f"{category}_test.jgz")
        try:
            with gzip.open(anno, "r") as fin:
                annotation = json.loads(fin.read())
        except FileNotFoundError:
            continue
        for seq_name in sorted(annotation.keys()):
            seq_data = annotation[seq_name]
            if len(seq_data) < num_frames:
                continue
            ids = np.random.choice(len(seq_data), num_frames, replace=False)
            names = [os.path.join(erq.CO3D_DIR, seq_data[i]["filepath"]) for i in ids]
            imgs = load_and_preprocess_images(names).to(device=device, dtype=dtype)
            # Resize to the engine's fixed (H,W) EXACTLY as TRTAllEngineModel.forward
            # does — otherwise native preprocessing yields a different token count P
            # than the fixed-shape engine expects (cause D).
            if tuple(imgs.shape[-2:]) != (tep.DEFAULT_H, tep.DEFAULT_W):
                imgs = torch.nn.functional.interpolate(
                    imgs, size=(tep.DEFAULT_H, tep.DEFAULT_W),
                    mode="bilinear", align_corners=False)
            return (imgs.unsqueeze(0) if imgs.dim() == 4 else imgs)
    raise RuntimeError("No CO3D calibration sequence found — check dataset paths.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="deploy/artifacts/engines/camera_head.trt")
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--zeros", action="store_true",
                   help="use a zeros input instead of real frames (no dataset needed)")
    args = p.parse_args()

    import eval_co3d_realquant as erq
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    full = erq._load_model(tep.DEFAULT_CKPT, torch.float16, dev)
    full.depth_head = full.point_head = full.track_head = None
    agg = full.aggregator

    if args.zeros:
        images = torch.zeros(1, args.frames, 3, tep.DEFAULT_H, tep.DEFAULT_W,
                             dtype=torch.float16, device=dev)
    else:
        images = _one_real_sequence(args.frames, dev, torch.float16)

    with torch.inference_mode(), tep._quiet_and_batched():
        tokens, pos, B, S, P, C = agg.embed_tokens(images)
        blocks = tep.AggregatorBlocks(agg, pos, B, S, P, C).eval()
        last_tokens = blocks(tokens)                       # [1,S,P,2C], torch fp16
        pose_pt = full.camera_head([last_tokens])[-1]      # [1,S,9], torch reference

    last_np = last_tokens.detach().float().cpu().numpy()
    pose_pt_np = pose_pt.detach().float().cpu().numpy()
    print(f"\nlast_tokens {tuple(last_np.shape)}  pose_enc {tuple(pose_pt_np.shape)}")
    print("=== fp16 dynamic range (torch reference) ===")
    _range("last_tokens", last_tokens)
    _range("pose_enc", pose_pt)

    # Free the heavy model (keep last_tokens/pose on CPU), defrag, then load the engine.
    full.camera_head = None
    agg.frame_blocks = None
    agg.global_blocks = None
    del blocks, agg, full, tokens, last_tokens, pose_pt
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
        try:
            blk = torch.empty(700 * 1024 * 1024 // 2, dtype=torch.float16, device=dev)
            del blk
        except Exception:
            pass
        torch.cuda.empty_cache()

    from deploy.runtime.trt_inference import TRTInferenceEngine
    print(f"\nrunning camera head in TensorRT ({os.path.basename(args.engine)}) ...")
    eng = TRTInferenceEngine(args.engine)
    in_name = [eng.engine.get_tensor_name(i) for i in range(eng.engine.num_io_tensors)
               if eng.engine.get_tensor_mode(eng.engine.get_tensor_name(i))
               == eng._trt.TensorIOMode.INPUT][0]
    pose_trt = eng.infer({in_name: last_np})[eng._output_names[0]]
    eng.close()

    print("\n=== camera_head: TRT vs PyTorch fp16 (pose_enc) ===")
    print(" ", _err(pose_trt, pose_pt_np))
    _range("pose_enc_trt", pose_trt)
    print("\nLarge pose_enc error here → the camera head is a dominant cause of the all-TRT "
          "collapse. |max| near 65504 on any tensor → fp16 overflow (cause B); otherwise "
          "fp16 precision in the head's regression trunk (cause A, localized to the head).")


if __name__ == "__main__":
    main()
