"""
INT8 calibration data loader for TensorRT PTQ.

Used when building a TRT engine from a non-QAT (FP32/FP16) checkpoint
without Q/DQ nodes — TRT needs calibration data to determine per-layer scales.

For QAT checkpoints (with Q/DQ nodes from onnx_exporter.py), calibration
is NOT needed: scales are baked into the ONNX graph. Use this only for the
ModelOpt Phase 2 PTQ path or for validating PTQ vs QAT accuracy.

Usage:
  from deploy.export.calibration import CO3DCalibrator
  calibrator = CO3DCalibrator(data_root, num_frames=4, num_batches=200)
  # Pass to trt_builder.build_engine(..., calibrator=calibrator)
"""

import numpy as np
import torch


class CO3DCalibrator:
    """
    TensorRT INT8 entropy calibrator backed by CO3D data.

    Feeds `num_batches` batches of shape (1, num_frames, 3, H, W) to TRT
    for calibration scale estimation (entropy minimization).
    """

    def __init__(
        self,
        data_root: str,
        num_frames: int = 4,
        image_size: int = 224,
        num_batches: int = 200,
        cache_file: str = 'deploy/engines/calibration.cache',
    ):
        self.num_frames = num_frames
        self.image_size = image_size
        self.num_batches = num_batches
        self.cache_file = cache_file
        self._batch_idx = 0
        self._data = self._load_co3d(data_root)

        # Pre-allocate CUDA buffer for calibration inputs
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
        batch_shape = (1, num_frames, 3, image_size, image_size)
        self._cuda_buf = cuda.mem_alloc(int(np.prod(batch_shape)) * 4)  # float32

    def _load_co3d(self, data_root: str):
        # Import your CO3D dataloader here and return an iterator/list of
        # pre-loaded tensors in shape (1, num_frames, 3, H, W), float32, [0,1].
        #
        # Placeholder: returns random tensors so the calibrator can be used
        # without a dataset for initial engine builds. Replace with real data.
        n = self.num_batches
        shape = (n, 1, self.num_frames, 3, self.image_size, self.image_size)
        return torch.rand(*shape).numpy().astype(np.float32)

    # TRT calibrator interface
    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self._batch_idx >= self.num_batches:
            return None
        import pycuda.driver as cuda
        batch = self._data[self._batch_idx]
        cuda.memcpy_htod(self._cuda_buf, np.ascontiguousarray(batch))
        self._batch_idx += 1
        return [int(self._cuda_buf)]

    def read_calibration_cache(self):
        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        if self.cache_file:
            import os
            os.makedirs(os.path.dirname(self.cache_file) or '.', exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                f.write(cache)
