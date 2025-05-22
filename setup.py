from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='triangle_mult_cuda',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'triangle_mult_cuda', [
                'triangle_mult.cpp',
                'triangle_mult_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3', 
                    '-arch=sm_80',
                    '--extended-lambda',
                    '--expt-relaxed-constexpr'
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)