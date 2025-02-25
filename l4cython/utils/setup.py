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
    include_dirs = ['/usr/include', '/usr/include/hdf', '/usr/lib'],
    extra_compile_args = [
        '-g1', '-Wall', '-fPIC', '-D_GNU_SOURCE', '-DHAVE_UUID', '-DHAVE_HDF4',
        '-ldfalt', '-lmfhdfalt', '-lz', '-lsz', '-luuid', '-lm', '-lutil'
    ]
)

hdf5 = Extension(
    name = 'hdf5',
    sources = ['hdf5.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'mfhdfalt', 'hdf5'],
    include_dirs = ['/usr/include', '/usr/include/hdf5/openmpi/', '/usr/lib', '/usr/lib/x86_64-linux-gnu/openmpi/include/'],
    extra_compile_args = [
        '-g1', '-Wall', '-fPIC', '-D_GNU_SOURCE', '-DHAVE_UUID', '-DHAVE_HDF4',
        '-ldfalt', '-lmfhdfalt', '-lhdf5', '-lz', '-lsz', '-luuid', '-lm', '-lutil'
    ],
    extra_link_args = [
        '-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/',
    ]
)

io = Extension(
    name = 'io',
    sources = ['io.pyx'],
    define_macros = MACROS
)

dec2bin = Extension(
    name = 'dec2bin',
    sources = ['dec2bin.pyx'],
    define_macros = MACROS
)

setup(ext_modules = cythonize([hdf5, io, mkgrid, dec2bin]))
