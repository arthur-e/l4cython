# cython: language_level=3

from distutils.core import setup, Extension
from Cython.Build import cythonize

MACROS = [ # Avoids warning "Using deprecated NumPy API"
    ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')
] # https://stackoverflow.com/questions/52749662/using-deprecated-numpy-api

# NOTE: Because libgdal-dev and libhdf4-dev conflict on Ubuntu,
#   it is necesary to install libhdf4-alt-dev if libgdal-dev is desired;
#   consequently, -ldf and -lmfhdf flags become -ldfalt and -lmfhdfalt
mkgrid = Extension(
    name = 'mkgrid',
    sources = ['mkgrid.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'mfhdfalt'],
    include_dirs = ['/usr/include', '/usr/include/hdf', 'usr/lib'],
    extra_compile_args = [
        '-g', '-Wall', '-fPIC', '-D_GNU_SOURCE', '-DHAVE_UUID', '-DHAVE_HDF4',
        '-ldfalt', '-lmfhdfalt', '-lz', '-lsz', '-luuid', '-lm', '-lutil'
    ]
)

setup(ext_modules = cythonize([mkgrid]))
