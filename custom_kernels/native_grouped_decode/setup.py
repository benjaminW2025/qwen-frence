"""Build the native grouped-GQA decode CUDA extension in place."""

from pathlib import Path
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


HERE = Path(__file__).resolve().parent
os.chdir(HERE)

setup(
    name="native-grouped-decode",
    ext_modules=[
        CUDAExtension(
            name="_native_grouped_decode",
            sources=[str(HERE / "native_grouped_decode.cu")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
