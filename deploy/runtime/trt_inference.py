"""
TensorRT inference wrapper for quantized VGGT.

Loads a serialized .trt engine, manages CUDA I/O buffers, and provides a
simple forward() interface compatible with the existing VGGT evaluation code.

Usage:
  from deploy.runtime.trt_inference import TRTInferenceEngine

  engine = TRTInferenceEngine('deploy/engines/vggt_w8a8.trt')
  output = engine.infer({'images': images_tensor})  # images: (1, N, 3, H, W)
  engine.close()
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch


class TRTInferenceEngine:
    """
    Minimal TensorRT inference engine wrapper.

    Manages:
    - Engine deserialization
    - CUDA memory allocation (input + output buffers)
    - Binding shape updates for dynamic axes (frames dimension)
    - Synchronous inference execution
    """

    def __init__(self, engine_path: str, device_index: int = 0):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError as e:
            raise ImportError(f'Missing TRT/PyCUDA dependency: {e}')

        self._trt = trt
        self._cuda = cuda

        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)

        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self._stream = cuda.Stream()
        self._buffers = {}   # name → (host_buf, device_buf)
        self._output_names = []

        # Identify outputs (all bindings that are not inputs)
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.OUTPUT:
                self._output_names.append(name)

        print(f'TRT engine loaded: {engine_path}')
        print(f'  Inputs:  {[self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors) if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]}')
        print(f'  Outputs: {self._output_names}')

    def _alloc_buffer(self, name: str, shape: tuple, dtype=np.float32):
        import pycuda.driver as cuda
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        host = cuda.pagelocked_empty(int(np.prod(shape)), dtype=dtype)
        device = cuda.mem_alloc(nbytes)
        self._buffers[name] = (host, device)

    def infer(self, inputs: dict) -> dict:
        """
        Run one inference pass.

        Args:
            inputs: dict mapping input tensor name to torch.Tensor or np.ndarray.
                    Tensors are automatically moved to CPU and converted to float32.

        Returns:
            dict mapping output tensor name to np.ndarray.
        """
        import pycuda.driver as cuda

        for name, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                arr = tensor.detach().cpu().float().numpy()
            else:
                arr = np.ascontiguousarray(tensor, dtype=np.float32)

            shape = arr.shape
            # Update dynamic shape binding for this input
            self.context.set_input_shape(name, shape)

            if name not in self._buffers or self._buffers[name][0].shape != arr.shape:
                self._alloc_buffer(name, shape)

            host_buf, dev_buf = self._buffers[name]
            np.copyto(host_buf, arr.ravel())
            cuda.memcpy_htod_async(dev_buf, host_buf, self._stream)
            self.context.set_tensor_address(name, int(dev_buf))

        # Allocate output buffers based on inferred shapes
        for name in self._output_names:
            out_shape = tuple(self.context.get_tensor_shape(name))
            if name not in self._buffers or self._buffers[name][0].shape[0] != int(np.prod(out_shape)):
                self._alloc_buffer(name, out_shape)
            _, dev_buf = self._buffers[name]
            self.context.set_tensor_address(name, int(dev_buf))

        # Execute
        self.context.execute_async_v3(stream_handle=self._stream.handle)
        self._stream.synchronize()

        # Copy outputs back to host
        outputs = {}
        for name in self._output_names:
            out_shape = tuple(self.context.get_tensor_shape(name))
            host_buf, dev_buf = self._buffers[name]
            cuda.memcpy_dtoh_async(host_buf, dev_buf, self._stream)
            self._stream.synchronize()
            outputs[name] = host_buf.reshape(out_shape).copy()

        return outputs

    def close(self):
        del self.context
        del self.engine

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ──────────────────────────────────────────────────────────────────────────────
# Quick validation: compare TRT output vs PyTorch (FP32) baseline
# ──────────────────────────────────────────────────────────────────────────────

def validate_trt_vs_pytorch(
    engine_path: str,
    pytorch_model: torch.nn.Module,
    num_frames: int = 4,
    image_size: int = 224,
    tol: float = 0.05,
):
    """
    Runs the same random input through PyTorch (FP32) and TRT, prints max/mean error.
    tol is the acceptable max absolute error as a fraction of output value range.
    """
    dummy = torch.randn(1, num_frames, 3, image_size, image_size)

    pytorch_model.eval()
    with torch.no_grad():
        pt_out = pytorch_model(images=dummy)
    if isinstance(pt_out, dict):
        pt_arr = next(iter(pt_out.values())).numpy()
    else:
        pt_arr = pt_out.numpy()

    with TRTInferenceEngine(engine_path) as engine:
        trt_out = engine.infer({'images': dummy})
    trt_arr = next(iter(trt_out.values()))

    abs_err = np.abs(pt_arr - trt_arr)
    val_range = pt_arr.max() - pt_arr.min() + 1e-8
    rel_err = abs_err / val_range
    print(f'Max abs error: {abs_err.max():.4f}  '
          f'Mean abs error: {abs_err.mean():.4f}  '
          f'Max rel error: {rel_err.max()*100:.2f}%')
    if rel_err.max() > tol:
        print(f'WARNING: max relative error {rel_err.max()*100:.2f}% > tolerance {tol*100:.1f}%')
    else:
        print('Validation PASSED.')
