from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sc_extension_cuda',
    ext_modules=[
        CUDAExtension('sc_extension_cuda', [
            'sc_cuda.cpp',
            'sc_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
