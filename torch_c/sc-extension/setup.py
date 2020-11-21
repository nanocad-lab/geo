from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

os.environ['CFLAGS'] = '-march=native -fopenmp'
# -fopenmp
setup(name='sc_extension',
      ext_modules=[CppExtension('sc_extension', ['sc.cpp'])],
#       extra_compile_args = ["-march=native"],
      cmdclass={'build_ext': BuildExtension})
