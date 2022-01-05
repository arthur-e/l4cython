# cython: language_level=3

from distutils.core import setup
from Cython.Build import cythonize

setup(name = 'reco', ext_modules = cythonize('reco.pyx'), language_level = 3)
