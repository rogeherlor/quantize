#!/usr/bin/env python3
"""Localize the TRT-chain accuracy loss.

Runs the SAME aggregator pairs [0:pairs) two ways on identical embed tokens —
(1) PyTorch (fp16), (2) TensorRT engine s0 — and reports the output error. A large
TRT-vs-PyTorch-fp16 error means the engine's all-fp16 compute (LayerNorm/softmax/RoPE)
is the loss, and the fix is a precision-aware export keeping those ops in fp32.

The PyTorch blocks are dropped (and NvMap defragged) before the engine loads, so both
the 2.4 GB model and the 0.4 GB engine never co-reside.

Usage (inside the Jetson container):
  python scripts/trt_numdiff.py
  python scripts/trt_numdiff.py --pairs 8 --engine deploy/artifacts/engines/chain_f10_c8_s0.trt
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="deploy/artifacts/engines/chain_f10_c8_s0.trt")
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--full", action="store_true",
                   help="compare the FULL 24-pair chain output vs PyTorch (all 3 chain engines)")
    p.add_argument("--chain", nargs="+",
                   default=["deploy/artifacts/engines/chain_f10_c8_s0.trt",
                            "deploy/artifacts/engines/chain_f10_c8_s8.trt",
                            "deploy/artifacts/engines/chain_f10_c8_s16.trt"])
    args = p.parse_args()

    import eval_co3d_realquant as erq
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    full = erq._load_model(tep.DEFAULT_CKPT, torch.float16, dev)
    full.depth_head = full.point_head = full.track_head = None
    agg = full.aggregator

    dummy = torch.zeros(1, args.frames, 3, tep.DEFAULT_H, tep.DEFAULT_W,
                        dtype=torch.float16, device=dev)
    with torch.inference_mode(), tep._quiet_and_batched():
        tokens, pos, B, S, P, C = agg.embed_tokens(dummy)
    if args.full:
        # PyTorch: ALL 24 pairs → output_list[-1] (the camera-head input [B,S,P,2C]).
        print(f"tokens {tuple(tokens.shape)}  running ALL 24 pairs in PyTorch fp16 ...")
        blocks = tep.AggregatorBlocks(agg, pos, B, S, P, C).eval()
        with torch.inference_mode(), tep._quiet_and_batched():
            pt_full = blocks(tokens).float().cpu().numpy()
        agg.frame_blocks = None
        agg.global_blocks = None
        del blocks
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()
            try:
                blk = torch.empty(1500 * 1024 * 1024 // 2, dtype=torch.float16, device=dev)
                del blk
            except Exception:
                pass
            torch.cuda.empty_cache()
        from deploy.runtime.trt_inference import TRTInferenceEngine
        engines = [TRTInferenceEngine(p) for p in args.chain]
        t = tokens
        for e in engines[:-1]:
            out = e.infer({"tokens": t})[e._output_names[0]]
            t = torch.from_numpy(out).to(dev, torch.float16)
        trt_full = engines[-1].infer({"tokens": t})[engines[-1]._output_names[0]]
        for e in engines:
            e.close()
        print("\n=== FULL chain (24 pairs) vs PyTorch fp16 ===")
        print(" ", _err(trt_full, pt_full))
        print("\nIf this is ~a few %% → the 1.3%%/chunk compounds (inherent TRT-vs-PyTorch "
              "fp16 kernels); if it is huge → a chain/handoff bug.")
        return

    print(f"tokens {tuple(tokens.shape)}  running pairs [0:{args.pairs}) in PyTorch fp16 ...")

    sl16 = tep.AggregatorBlocksSlice(agg, pos, B, S, P, C, 0, args.pairs, is_last=False).eval()
    with torch.inference_mode(), tep._quiet_and_batched():
        pt16 = sl16(tokens).float().cpu().numpy()

    # Free the heavy block weights (keep `tokens`), defrag, then load the engine.
    agg.frame_blocks = None
    agg.global_blocks = None
    del sl16
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
        try:
            blk = torch.empty(600 * 1024 * 1024 // 2, dtype=torch.float16, device=dev)
            del blk
        except Exception:
            pass
        torch.cuda.empty_cache()

    from deploy.runtime.trt_inference import TRTInferenceEngine
    print(f"running pairs [0:{args.pairs}) in TensorRT ({os.path.basename(args.engine)}) ...")
    eng = TRTInferenceEngine(args.engine)
    trt = eng.infer({"tokens": tokens})[eng._output_names[0]]
    eng.close()

    print("\n=== TRT vs PyTorch fp16 (pairs 0:%d) ===" % args.pairs)
    print(" ", _err(trt, pt16))
    print("\nLarge error (rel_mean ≫ 1%%) → the engine's all-fp16 compute is the loss; "
          "fix = precision-aware export keeping LayerNorm/RoPE in fp32.")


if __name__ == "__main__":
    main()
