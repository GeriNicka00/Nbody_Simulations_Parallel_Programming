# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 15:29:45 2022

@author: Geri Nicka
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "Nbody_OMP",
        ["Nbody_OMP.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-lgomp', '-Wl,-rpath,/usr/local/opt/gcc/lib/gcc/11/'],
    )
]

setup(name="Nbody_OMP",include_dirs=[numpy.get_include()],
      ext_modules=cythonize(ext_modules))

