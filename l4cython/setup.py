# cython: language_level=3

# NOTE: Removed these link args, seemed extraneous, but if OpenMP stops linking:
#   extra_link_args = ['-fopenmp', 

from distutils.core import setup, Extension
from Cython.Build import cythonize

MACROS = [ # Avoids warning "Using deprecated NumPy API"
    ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'), # https://stackoverflow.com/questions/52749662/using-deprecated-numpy-api
]
HDF4_DIRS = ['/usr/include', '/usr/include/hdf']
HDF4_ARGS = [
    '-Wall', '-fPIC', '-D_GNU_SOURCE', '-DHAVE_UUID', '-DHAVE_HDF4',
    '-ldfalt', '-lmfhdfalt', '-luuid',
]
HDF5_DIRS = [
    *HDF4_DIRS,
    '/usr/include/hdf5/openmpi/', # Also required for hdf5.pxd
    '/usr/lib/x86_64-linux-gnu/openmpi/include/' # Also required for hdf5.pxd
]
HDF5_LINKS = ['-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/']
# NOTE: For profiling, add the following to MACROS:
#   ('CYTHON_NOGIL_TRACE', '1') # https://cython.readthedocs.io/en/latest/src/tutorial/profiling_tutorial.html

budget = Extension(
    name = 'budget',
    sources = ['budget.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'hdf5'],
    include_dirs = ['./utils', *HDF5_DIRS],
    extra_compile_args = ['-Wno-maybe-uninitialized'],
    extra_link_args = [*HDF5_LINKS]
)

core = Extension(
    name = 'core',
    sources = ['core.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'mfhdfalt'],
    include_dirs = HDF4_DIRS,
    extra_compile_args = HDF4_ARGS
)

gpp = Extension(
    name = 'gpp',
    sources = ['gpp.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'hdf5'],
    include_dirs = ['./utils', *HDF5_DIRS],
    extra_compile_args = ['-Wno-maybe-uninitialized'],
    extra_link_args = [*HDF5_LINKS]
)

reco = Extension(
    name = 'reco',
    sources = ['reco.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'hdf5'],
    include_dirs = ['./utils', *HDF5_DIRS],
    extra_compile_args = ['-Wno-maybe-uninitialized'],
    extra_link_args = [*HDF5_LINKS]
)

resample = Extension(
    name = 'resample',
    sources = ['resample.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'hdf5'],
    include_dirs = ['./utils', *HDF5_DIRS],
    extra_compile_args = [
        '-DHAVE_HDF4', '-ldfalt', '-lhdf5', '-Wno-maybe-uninitialized', '-lz'
    ],
    extra_link_args = [*HDF5_LINKS]
)

spinup = Extension(
    name = 'spinup',
    sources = ['spinup_9km.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'mfhdfalt'],
    include_dirs = ['./utils', *HDF4_DIRS],
    extra_compile_args = HDF4_ARGS
)

setup(ext_modules = cythonize([core, resample, budget, gpp, reco, budget, spinup]))
