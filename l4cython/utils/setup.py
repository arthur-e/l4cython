# cython: language_level=3

from distutils.core import setup, Extension
from Cython.Build import cythonize

MACROS = [ # Avoids warning "Using deprecated NumPy API"
    ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')
] # https://stackoverflow.com/questions/52749662/using-deprecated-numpy-api

mkgrid = Extension(
    name = 'mkgrid',
    sources = ['mkgrid.pyx'],
    define_macros = MACROS,
    libraries = ['df', 'mfhdf'],
    include_dirs = ['/usr/include', '/usr/include/hdf'],
    extra_compile_args = [
        '-g', '-Wall', '-fPIC', '-D_GNU_SOURCE', '-DHAVE_UUID', '-DHAVE_HDF4',
        '-ldf', '-lmfhdf', '-lz', '-lsz', '-luuid', '-lm', '-lutil'
    ]
)

setup(ext_modules = cythonize([mkgrid]))
