"""Build the C++ scheduler extension without requiring Ninja."""

from pathlib import Path
import sys
import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


HERE = Path(__file__).resolve().parent

# Object timestamps do not track changes to the installed Torch wheel/ABI.
class SchedulerBuildExtension(BuildExtension):
    def finalize_options(self):
        super().finalize_options()
        self.force = True


compile_args = ["-O3", "-std=c++17"]
link_args = []
if sys.platform.startswith("linux"):
    compile_args.append(f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}")
elif sys.platform == "darwin":
    # The extension links Torch dylibs via @rpath. A direct `build_ext`
    # (rather than a wheel install) needs the Torch library directory embedded
    # so the CPU-only scheduler gates can import the freshly built module.
    link_args.append(f"-Wl,-rpath,{Path(torch.__file__).resolve().parent / 'lib'}")


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
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
    ],
    cmdclass={"build_ext": SchedulerBuildExtension.with_options(use_ninja=False)},
)
