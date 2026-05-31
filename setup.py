from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

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
