"""Build the C++ scheduler extension without requiring Ninja."""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


HERE = Path(__file__).resolve().parent


setup(
    name="inference-engine-cpp",
    ext_modules=[
        CppExtension(
            name="inference_engine_cpp",
            sources=[
                str(HERE / "src" / "iteration_loop.cpp"),
                str(HERE / "src" / "batch_builder.cpp"),
                str(HERE / "src" / "bindings.cpp"),
            ],
            include_dirs=[str(HERE / "include")],
            extra_compile_args=["-O3", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
