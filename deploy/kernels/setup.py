"""
Builds the INT4 GEMM CUDA extension.

Usage:
  cd /home/rogelio/quantize
  python deploy/kernels/setup.py build_ext --inplace

The compiled .so is placed in deploy/kernels/ and importable as:
  from deploy.kernels import int4_gemm_ext

Prerequisites:
  - CUDA 12.6 toolkit (nvcc in PATH)
  - PyTorch with CUDA (Jetson wheel)
  - CUTLASS 3.x at /opt/cutlass (for the real INT4 kernel; not needed for the
    current fallback which uses torch._int_mm)
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

CUTLASS_ROOT = os.environ.get('CUTLASS_ROOT', '/opt/cutlass')

extra_include = [os.path.join(CUTLASS_ROOT, 'include')]
extra_compile_args = {
    'cxx': ['-O3'],
    'nvcc': [
        '-O3',
        '--generate-code', 'arch=compute_87,code=sm_87',  # Jetson Orin Nano
        '-U__CUDA_NO_HALF_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__',
    ],
}

setup(
    name='int4_gemm_ext',
    ext_modules=[
        CUDAExtension(
            name='deploy.kernels.int4_gemm_ext',
            sources=['deploy/kernels/int4_gemm.cu'],
            include_dirs=extra_include,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
