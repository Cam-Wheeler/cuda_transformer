import os
from setuptools import setup
from pathlib import Path
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

SETUP_DIR = Path(__file__).parent

# Without a visible GPU (I am building on mac), torch can't detect archs
# and build crashes with IndexError in _get_cuda_arch_flags. 
# So we are going to set the env variable to an A100 = 8.0.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")

setup(
    name='training_cuda',
    packages=[],
    ext_modules=[
        CUDAExtension(
            name='custom_training',
            sources=[
                str(SETUP_DIR / 'cudasrc' / 'training' / 'binding.cpp'),
                str(SETUP_DIR / 'cudasrc' / 'training' / 'kernels' / "elementwise.cu"),
            ],
            extra_compile_args={
                "cxx": ["-g"],
                "nvcc": ["-O2"]
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)