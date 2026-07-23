from setuptools import setup
from pathlib import Path
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

SETUP_DIR = Path(__file__).parent

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