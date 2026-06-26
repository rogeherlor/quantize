#!/usr/bin/env python3
"""Offline fp32 → fp16 ONNX converter — uses ONLY `onnx` (no onnxconverter-common,
which can't be pip-installed in the offline Jetson container).

Why: TensorRT's builder kernel library alone takes ~3 GB GPU on the Orin Nano, so an
fp32 network (~2.3 GB of weights) leaves no room for the attention tactic and the build
OOMs (Error Code 10). Halving the weights to fp16 (~1.15 GB) frees ~1.15 GB so the
memory-frugal attention tactic fits.

Approach: cast every FLOAT initializer / Constant / Cast-target to FLOAT16 and set the
graph inputs+outputs to FLOAT16, i.e. make the WHOLE graph fp16. No Cast-node insertion
(the brittle part of onnxconverter-common) — TRT builds it directly with --fp16, and the
hybrid runtime (TRTInferenceEngine.infer) already casts tokens to the engine's I/O dtype,
so nothing downstream changes. The final engine would be fp16 either way; this only
shrinks the BUILD-TIME footprint.

Usage:
  python scripts/onnx_to_fp16.py \
      --in  deploy/artifacts/onnx/pose_baseline/pose_baseline.onnx \
      --out deploy/artifacts/onnx/pose_fp16h/pose_fp16h.onnx
"""
import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

FP32 = TensorProto.FLOAT
FP16 = TensorProto.FLOAT16


def _cast_tensor(t):
    """Cast a FLOAT TensorProto in place to FLOAT16; return True if changed."""
    if t.data_type != FP32:
        return False
    arr = numpy_helper.to_array(t).astype(np.float16)
    t.CopyFrom(numpy_helper.from_array(arr, t.name))
    return True


def convert(src, dst):
    model = onnx.load(src)  # pulls in external-data weights
    g = model.graph

    n_init = sum(_cast_tensor(init) for init in g.initializer)

    n_const = 0
    n_cast = 0
    for node in g.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == FP32:
                    n_const += _cast_tensor(attr.t)
        elif node.op_type == "Cast":
            # a Cast that targets fp32 would re-introduce fp32 mid-graph → keep it fp16
            for attr in node.attribute:
                if attr.name == "to" and attr.i == FP32:
                    attr.i = FP16
                    n_cast += 1

    # Graph I/O → fp16 (only the FLOAT ones; leave int/bool inputs alone).
    for vi in list(g.input) + list(g.output):
        tt = vi.type.tensor_type
        if tt.elem_type == FP32:
            tt.elem_type = FP16

    # Drop stale intermediate type/shape hints so TRT re-infers fp16 throughout.
    del g.value_info[:]

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    onnx.save(model, dst, save_as_external_data=True,
              all_tensors_to_one_file=True, location=os.path.basename(dst) + ".data")
    print(f"fp16 conversion: {n_init} initializers, {n_const} constants, "
          f"{n_cast} Cast targets → fp16; I/O set to fp16. Wrote {dst}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--out", dest="dst", required=True)
    args = p.parse_args()
    convert(args.src, args.dst)


if __name__ == "__main__":
    main()
