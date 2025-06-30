# cython: language_level=3

from distutils.core import setup, Extension
from Cython.Build import cythonize

MACROS = [ # Avoids warning "Using deprecated NumPy API"
    ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')
] # https://stackoverflow.com/questions/52749662/using-deprecated-numpy-api
HDF4_DIRS = ['/usr/include', '/usr/include/hdf']
HDF4_ARGS = [
    '-Wall', '-fPIC', '-D_GNU_SOURCE', '-DHAVE_UUID', '-DHAVE_HDF4',
    '-ldfalt', '-lmfhdfalt', '-luuid',
]
HDF5_DIRS = [
    *HDF4_DIRS,
    '/usr/include/hdf5/openmpi/', # Also required for hdf5.pxd
    '/usr/include/hdf5/serial/',
    '/usr/lib/x86_64-linux-gnu/openmpi/include/' # Also required for hdf5.pxd
]
HDF5_LINKS = ['-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/']
# NOTE: For profiling, add the following to MACROS:
#   ('CYTHON_NOGIL_TRACE', '1') # https://cython.readthedocs.io/en/latest/src/tutorial/profiling_tutorial.html


# l4cython/utils/ ##############################################################

# NOTE: Because libgdal-dev and libhdf4-dev conflict on Ubuntu,
#   it is necesary to install libhdf4-alt-dev if libgdal-dev is desired;
#   consequently, -ldf and -lmfhdf flags become -ldfalt and -lmfhdfalt
mkgrid = Extension(
    name = 'l4cython.utils.mkgrid',
    sources = ['l4cython/utils/mkgrid.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'mfhdfalt'],
    include_dirs = ['.', *HDF4_DIRS],
    extra_compile_args = [
        '-g1', '-Wall', '-fPIC', '-D_GNU_SOURCE', '-DHAVE_UUID', '-DHAVE_HDF4',
        '-ldfalt', '-lmfhdfalt', '-lz', '-luuid', '-lm', '-lutil'
    ]
)

hdf5 = Extension(
    name = 'l4cython.utils.hdf5',
    sources = ['l4cython/utils/hdf5.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'mfhdfalt', 'hdf5'],
    include_dirs = ['.', *HDF5_DIRS],
    extra_compile_args = [
        '-g1', '-Wall', '-fPIC', '-D_GNU_SOURCE', '-DHAVE_UUID', '-DHAVE_HDF4',
        '-ldfalt', '-lmfhdfalt', '-lhdf5', '-lz', '-luuid', '-lm', '-lutil'
    ],
    extra_link_args = ['-fopenmp', *HDF5_LINKS]
)

io = Extension(
    name = 'l4cython.utils.io',
    sources = ['l4cython/utils/io.pyx'],
    define_macros = MACROS,
    include_dirs = ['.'],
)

dec2bin = Extension(
    name = 'l4cython.utils.dec2bin',
    sources = ['l4cython/utils/dec2bin.pyx'],
    define_macros = MACROS,
    include_dirs = ['.'],
)


# l4cython/ ##################################################################

budget = Extension(
    name = 'l4cython.budget',
    sources = ['l4cython/budget.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'hdf5'],
    include_dirs = ['.', './l4cython', './l4cython/utils', *HDF5_DIRS],
    extra_compile_args = ['-Wno-maybe-uninitialized'],
    extra_link_args = [*HDF5_LINKS]
)

core = Extension(
    name = 'l4cython.core',
    sources = ['l4cython/core.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'mfhdfalt'],
    include_dirs = ['.', *HDF4_DIRS],
    extra_compile_args = HDF4_ARGS
)

gpp = Extension(
    name = 'l4cython.gpp',
    sources = ['l4cython/gpp.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'hdf5'],
    include_dirs = ['.', './l4cython', './l4cython/utils', *HDF5_DIRS],
    extra_compile_args = ['-Wno-maybe-uninitialized'],
    extra_link_args = [*HDF5_LINKS]
)

reco = Extension(
    name = 'l4cython.reco',
    sources = ['l4cython/reco.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'hdf5'],
    include_dirs = ['.', './l4cython', './l4cython/utils', *HDF5_DIRS],
    extra_compile_args = ['-Wno-maybe-uninitialized'],
    extra_link_args = [*HDF5_LINKS]
)

resample = Extension(
    name = 'l4cython.resample',
    sources = ['l4cython/resample.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'hdf5'],
    include_dirs = ['.', './l4cython', './l4cython/utils', *HDF5_DIRS],
    extra_compile_args = [
        '-DHAVE_HDF4', '-ldfalt', '-lhdf5', '-Wno-maybe-uninitialized', '-lz'
    ],
    extra_link_args = [*HDF5_LINKS]
)

restart = Extension(
    name = 'l4cython.restart',
    sources = ['l4cython/restart.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'hdf5'],
    include_dirs = ['.', './l4cython', './l4cython/utils', *HDF5_DIRS],
    extra_compile_args = ['-Wno-maybe-uninitialized'],
    extra_link_args = [*HDF5_LINKS]
)

spinup = Extension(
    name = 'l4cython.spinup',
    sources = ['l4cython/spinup_9km.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'mfhdfalt'],
    include_dirs = ['.', './l4cython', './l4cython/utils', *HDF4_DIRS],
    extra_compile_args = HDF4_ARGS
)

setup(
    name = 'l4cython',
    packages = ['l4cython', 'l4cython.utils'],
    ext_modules = cythonize([
        hdf5, io, mkgrid, dec2bin, # l4cython/utils
        core, resample, restart, budget, gpp, reco, spinup
    ]))
