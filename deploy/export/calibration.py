"""
INT8 calibration data loader for TensorRT automatic PTQ.

Used when building a TRT engine from a non-QDQ (FP32/FP16) ONNX graph — TRT needs
real calibration data to estimate per-tensor INT8 scales (entropy minimization).

For QAT/LSQ checkpoints exported with Q/DQ nodes (onnx_exporter.py / trt_export_pose.py
--precision w8a8), calibration is NOT needed: the learned scales are baked into the
graph. Use this only for the "TensorRT automatic quantization" comparison point.

Usage:
  from deploy.export.calibration import CO3DCalibrator
  calib = CO3DCalibrator(num_frames=10, height=350, width=518, num_batches=64)
  build_engine(..., precision='int8', calibrator=calib, restrict_int8_to=QLAYERS)
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

try:
    import tensorrt as trt
    _CALIB_BASE = trt.IInt8EntropyCalibrator2
except ImportError:  # allow import on a host without TRT (e.g. for linting)
    trt = None
    _CALIB_BASE = object

# Mirror the dataset paths used by scripts/eval_co3d_realquant.py.
CO3D_DIR = "data3/rogelio/co3d/dataset/"
CO3D_ANNO_DIR = "data3/rogelio/co3d/preprocessed_dataset"
CALIB_CATEGORIES = ["apple", "backpack", "banana", "bench", "bicycle"]


class CO3DCalibrator(_CALIB_BASE):
    """TensorRT INT8 entropy calibrator backed by real CO3D frames.

    The blocks-only engine's input is `tokens` (not images), so this calibrator
    loads real CO3D sequences, runs the PyTorch DINO embed (`embed_fn`, i.e.
    Aggregator.embed_tokens) to turn each (1,F,3,H,W) image batch into the engine's
    `tokens` [F, P, C], and feeds those tokens to TRT — matching the engine binding
    exactly so the INT8 entropy scales are estimated on realistic block inputs.
    """

    def __init__(
        self,
        embed_fn,
        num_frames: int = 10,
        height: int = 350,
        width: int = 518,
        num_batches: int = 64,
        cache_file: str = "deploy/engines/calibration.cache",
        co3d_dir: str = CO3D_DIR,
        anno_dir: str = CO3D_ANNO_DIR,
    ):
        super().__init__()
        self.embed_fn = embed_fn
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.num_batches = num_batches
        self.cache_file = cache_file
        self.co3d_dir = co3d_dir
        self.anno_dir = anno_dir
        self._batch_idx = 0

        import pycuda.driver as cuda
        try:
            import pycuda.autoprimaryctx  # noqa: F401  (share torch's primary ctx)
        except ImportError:
            import pycuda.autoinit  # noqa: F401
        self._data = self._load_co3d()              # list of token numpy arrays
        self.embed_fn = None                        # drop ref so the torch model can be freed
        self._nbytes = int(self._data[0].nbytes)    # one token batch [F, P, C] float32
        self._cuda_buf = cuda.mem_alloc(self._nbytes)

    def _load_co3d(self):
        """Load real CO3D sequences and embed them → list of token numpy arrays
        ([F, P, C] float32, the engine's `tokens` binding).

        Falls back to a single random batch (with a warning) if the dataset is not
        found, so an engine can still be built for plumbing tests.
        """
        import gzip
        import json
        from vggt.utils.load_fn import load_and_preprocess_images

        img_batches = []
        for category in CALIB_CATEGORIES:
            if len(img_batches) >= self.num_batches:
                break
            anno = os.path.join(self.anno_dir, f"{category}_test.jgz")
            try:
                with gzip.open(anno, "r") as fin:
                    annotation = json.loads(fin.read())
            except FileNotFoundError:
                continue
            for seq_name in sorted(annotation.keys()):
                if len(img_batches) >= self.num_batches:
                    break
                seq = annotation[seq_name]
                if len(seq) < self.num_frames:
                    continue
                ids = np.random.choice(len(seq), self.num_frames, replace=False)
                names = [os.path.join(self.co3d_dir, seq[i]["filepath"]) for i in ids]
                try:
                    imgs = load_and_preprocess_images(names)  # (F,3,H,W) float32
                except Exception:
                    continue
                # Resize to the engine's fixed spatial shape, add batch dim.
                imgs = F.interpolate(imgs, size=(self.height, self.width),
                                     mode="bilinear", align_corners=False)
                img_batches.append(imgs.unsqueeze(0).contiguous())  # torch (1,F,3,H,W)
                break  # one sequence per category per pass

        if not img_batches:
            print("WARNING: no CO3D calibration data found — using ONE random batch. "
                  "INT8 PTQ scales will be meaningless. Check dataset paths.")
            img_batches = [torch.rand(1, self.num_frames, 3, self.height, self.width)]

        # Embed images → tokens (the engine's actual input) via the PyTorch DINO stage.
        token_batches = []
        for imgs in img_batches:
            tokens = self.embed_fn(imgs)            # [F, P, C] tensor
            token_batches.append(
                np.ascontiguousarray(tokens.detach().cpu().float().numpy().astype(np.float32)))
        self.num_batches = len(token_batches)
        print(f"CO3DCalibrator: {self.num_batches} token calibration batches, "
              f"shape {token_batches[0].shape}.")
        return token_batches

    # ── TRT calibrator interface ──────────────────────────────────────────────
    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self._batch_idx >= self.num_batches:
            return None
        import pycuda.driver as cuda
        batch = np.ascontiguousarray(self._data[self._batch_idx])
        cuda.memcpy_htod(self._cuda_buf, batch)
        self._batch_idx += 1
        return [int(self._cuda_buf)]

    def read_calibration_cache(self):
        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        if self.cache_file:
            os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
            with open(self.cache_file, "wb") as f:
                f.write(cache)
