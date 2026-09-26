"""Build the independent SM90a attention candidate with CUDA 12.8+ and CuTe headers."""
import os
import hashlib
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cutlass = Path(os.environ.get('CUTLASS_PATH', '/nonexistent'))
if not (cutlass / 'include/cute/tensor.hpp').is_file():
    raise RuntimeError('Set CUTLASS_PATH to a CUTLASS v3.9.2 checkout (CuTe primitives only)')

setup(name='inference_hopper_attention', ext_modules=[CUDAExtension(
    'inference_hopper_attention', [str(Path(__file__).with_name('attention.cu'))],
    include_dirs=[str(cutlass / 'include')], libraries=['cuda'],
    define_macros=[('HOPPER_SOURCE_HASH', '"' + hashlib.sha256(
        Path(__file__).with_name('attention.cu').read_bytes()).hexdigest() + '"')],
    extra_compile_args={'cxx': ['-O3', '-std=c++17'],
                        'nvcc': ['-O3', '-std=c++17', '--expt-relaxed-constexpr',
                                 '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo']})],
    cmdclass={'build_ext': BuildExtension})
