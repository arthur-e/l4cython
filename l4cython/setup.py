# cython: language_level=3

from distutils.core import setup, Extension
from Cython.Build import cythonize

respiration = Extension(
    name = 'respiration',
    sources = ['respiration.pyx'],
    define_macros = [ # Avoids warning "Using deprecated NumPy API"
        ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')
    ], # https://stackoverflow.com/questions/52749662/using-deprecated-numpy-api
    extra_compile_args = ['-g'],
    extra_link_args = ['-g']
)

setup(ext_modules = cythonize(respiration, gdb_debug = True))
