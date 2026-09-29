"""Build the SM90a GEMM-epilogue kernels against a CUTLASS v3.9.2 checkout."""
import hashlib
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cutlass = Path(os.environ.get('CUTLASS_PATH', '/nonexistent'))
if not (cutlass / 'include/cutlass/gemm/collective/collective_builder.hpp').is_file() or \
        not (cutlass / 'tools/util/include/cutlass/util/packed_stride.hpp').is_file():
    raise RuntimeError('Set CUTLASS_PATH to a full CUTLASS v3.9.2 checkout (include/ and tools/util/include/)')

source = Path(__file__).with_name('fused_gemm.cu')
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

# GEMM_EPILOGUE_SWEEP=1 compiles every tile/cluster/scheduler candidate for
# experiments/prefill/sweep_epilogue_tiles.py; the default build compiles only the
# two defaults. Rebuild without it afterwards.
macros = [('GEMM_EPILOGUE_SOURCE_HASH', '"' + source_hash + '"')]
if os.environ.get('GEMM_EPILOGUE_SWEEP') == '1':
    macros.append(('GEMM_EPILOGUE_SWEEP', '1'))

setup(name='inference_gemm_epilogue', ext_modules=[CUDAExtension(
    'inference_gemm_epilogue', [str(source)],
    include_dirs=[str(cutlass / 'include'), str(cutlass / 'tools/util/include')],
    define_macros=macros,
    extra_compile_args={'cxx': ['-O3', '-std=c++17'],
                        'nvcc': ['-O3', '-std=c++17', '--expt-relaxed-constexpr',
                                 '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo']})],
    cmdclass={'build_ext': BuildExtension})
