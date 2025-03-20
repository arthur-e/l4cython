# cython: language_level=3

from distutils.core import setup, Extension
from Cython.Build import cythonize

MACROS = [ # Avoids warning "Using deprecated NumPy API"
    ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')
] # https://stackoverflow.com/questions/52749662/using-deprecated-numpy-api
HDF4_DIRS = ['/usr/include', '/usr/include/hdf']
HDF5_DIRS = [
    *HDF4_DIRS,
    '/usr/include/hdf5/openmpi/', # Also required for hdf5.pxd
    '/usr/lib/x86_64-linux-gnu/openmpi/include/' # Also required for hdf5.pxd
]

# NOTE: Because libgdal-dev and libhdf4-dev conflict on Ubuntu,
#   it is necesary to install libhdf4-alt-dev if libgdal-dev is desired;
#   consequently, -ldf and -lmfhdf flags become -ldfalt and -lmfhdfalt
mkgrid = Extension(
    name = 'mkgrid',
    sources = ['mkgrid.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'mfhdfalt'],
    include_dirs = HDF4_DIRS,
    extra_compile_args = [
        '-g1', '-Wall', '-fPIC', '-D_GNU_SOURCE', '-DHAVE_UUID', '-DHAVE_HDF4',
        '-ldfalt', '-lmfhdfalt', '-lz', '-luuid', '-lm', '-lutil'
    ]
)

hdf5 = Extension(
    name = 'hdf5',
    sources = ['hdf5.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'mfhdfalt', 'hdf5'],
    include_dirs = HDF5_DIRS,
    extra_compile_args = [
        '-g1', '-Wall', '-fPIC', '-D_GNU_SOURCE', '-DHAVE_UUID', '-DHAVE_HDF4',
        '-ldfalt', '-lmfhdfalt', '-lhdf5', '-lz', '-luuid', '-lm', '-lutil'
    ],
    extra_link_args = ['-fopenmp', '-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/']
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
