# cython: language_level=3

from distutils.core import setup, Extension
from Cython.Build import cythonize

MACROS = [ # Avoids warning "Using deprecated NumPy API"
    ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'), # https://stackoverflow.com/questions/52749662/using-deprecated-numpy-api
]
HDF_DIRS = [
    '/usr/include', '/usr/include/hdf',
    '/usr/include/hdf5/openmpi/', # Also required for hdf5.pxd
    '/usr/lib/x86_64-linux-gnu/openmpi/include/' # Also required for hdf5.pxd
]
# NOTE: For profile, add the following to MACROS:
#   ('CYTHON_NOGIL_TRACE', '1') # https://cython.readthedocs.io/en/latest/src/tutorial/profiling_tutorial.html

budget = Extension(
    name = 'budget',
    sources = ['budget.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'hdf5'],
    include_dirs = ['./utils', *HDF_DIRS],
    extra_compile_args = [
        '-fopenmp', '-DHAVE_HDF4', '-ldfalt', '-lhdf5', '-Wno-maybe-uninitialized', '-g1'
    ],
    extra_link_args = ['-fopenmp', '-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/']
)

gpp = Extension(
    name = 'gpp',
    sources = ['gpp.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'hdf5'],
    include_dirs = ['./utils', *HDF_DIRS],
    extra_compile_args = [
        '-fopenmp', '-DHAVE_HDF4', '-ldfalt', '-lhdf5', '-Wno-maybe-uninitialized', '-g1'
    ],
    extra_link_args = ['-fopenmp', '-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/']
)

reco = Extension(
    name = 'reco',
    sources = ['reco.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'hdf5'],
    include_dirs = ['./utils', *HDF_DIRS],
    extra_compile_args = [
        '-fopenmp', '-DHAVE_HDF4', '-ldfalt', '-lhdf5', '-Wno-maybe-uninitialized', '-g1'
    ],
    extra_link_args = ['-fopenmp', '-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/']
)

spinup = Extension(
    name = 'spinup',
    sources = ['spinup_9km.pyx'],
    define_macros = MACROS,
    extra_compile_args = ['-fopenmp', '-g1'],
    extra_link_args = ['-fopenmp']
)

resample = Extension(
    name = 'resample',
    sources = ['resample.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'hdf5'],
    include_dirs = ['./utils', *HDF_DIRS],
    extra_compile_args = [
        '-fopenmp', '-DHAVE_HDF4', '-ldfalt', '-lhdf5', '-Wno-maybe-uninitialized', '-g1'
    ],
    extra_link_args = ['-fopenmp', '-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/']
)

setup(ext_modules = cythonize([budget, gpp, reco, spinup, resample]))
