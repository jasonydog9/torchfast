from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension
import torch

extra_compile_args = ["-O3", "-march=native", "-ffast-math"]

extensions = [
    CppExtension(
        name="fused_linear_act",
        sources=["csrc/fused_linear_act.cpp"],
        extra_compile_args=extra_compile_args,
    ),
    CppExtension(
        name="fast_normalization",
        sources=["csrc/normalization.cpp"],
        extra_compile_args=extra_compile_args,
    ),
    CppExtension(
        name="fast_attention",
        sources=["csrc/attention.cpp"],
        extra_compile_args=extra_compile_args,
    ),
    CppExtension(
        name="focal_loss",
        sources=["csrc/losses.cpp"],
        extra_compile_args=extra_compile_args,
    ),
]

# CUDA extension — only built when NVIDIA GPU + CUDA toolkit are available.
# On macOS/CPU-only machines this is skipped gracefully.
# --use_fast_math enables: hardware rsqrtf, fused multiply-add, fast transcendentals.
if torch.cuda.is_available():
    from torch.utils.cpp_extension import CUDAExtension
    extensions.append(
        CUDAExtension(
            name="fast_normalization_cuda",
            sources=["csrc/normalization_cuda.cu"],
            extra_compile_args={
                "cxx":  ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    )

# MPS extension — Apple Silicon GPU via Metal (macOS only).
# .mm = Objective-C++: mixes C++ (PyTorch/pybind11) with Objective-C (Metal API).
# -fobjc-arc: enable Automatic Reference Counting for Objective-C objects.
# Framework links: Metal (GPU compute), MPS (optimized ML primitives), Foundation (NSString).
import sys
if sys.platform == "darwin" and torch.backends.mps.is_available():
    extensions.append(
        CppExtension(
            name="fast_normalization_mps",
            sources=["csrc/normalization_mps.mm"],
            extra_compile_args=["-O3", "-fobjc-arc"],
            extra_link_args=[
                "-framework", "Metal",
                "-framework", "MetalPerformanceShaders",
                "-framework", "MetalPerformanceShadersGraph",
                "-framework", "Foundation",
            ],
        )
    )

setup(
    name="torchfast",
    version="0.1.0",
    description="Custom C++ PyTorch kernels measurably faster than PyTorch defaults",
    author="Jason Yap",
    packages=find_packages(),
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=["torch>=2.0"],
)
