# cython: language_level=3

from distutils.core import setup, Extension
from Cython.Build import cythonize

respiration = Extension(
    name = 'reco',
    sources = ['reco_9km.pyx'],
    define_macros = [ # Avoids warning "Using deprecated NumPy API"
        ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')
    ] # https://stackoverflow.com/questions/52749662/using-deprecated-numpy-api
)

setup(ext_modules = cythonize(respiration))
