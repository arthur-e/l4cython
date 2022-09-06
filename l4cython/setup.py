# cython: language_level=3

from distutils.core import setup, Extension
from Cython.Build import cythonize

MACROS = [ # Avoids warning "Using deprecated NumPy API"
    ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')
] # https://stackoverflow.com/questions/52749662/using-deprecated-numpy-api

test = Extension(
    name = 'test', sources = ['reco_9km.pyx'], define_macros = MACROS
)

reco = Extension(
    name = 'reco', sources = ['reco.pyx'], define_macros = MACROS
)

spinup = Extension(
    name = 'spinup',
    sources = ['spinup_9km.pyx'],
    define_macros = MACROS,
    extra_compile_args = ['-fopenmp'],
    extra_link_args = ['-fopenmp']
)

setup(ext_modules = cythonize([reco, spinup, test]))
