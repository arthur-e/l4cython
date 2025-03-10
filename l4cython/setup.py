# cython: language_level=3

from distutils.core import setup, Extension
from Cython.Build import cythonize

MACROS = [ # Avoids warning "Using deprecated NumPy API"
    ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'), # https://stackoverflow.com/questions/52749662/using-deprecated-numpy-api
]
HDF4_DIRS = ['/usr/include', '/usr/include/hdf']
HDF5_DIRS = [
    *HDF4_DIRS,
    '/usr/include/hdf5/openmpi/', # Also required for hdf5.pxd
    '/usr/lib/x86_64-linux-gnu/openmpi/include/' # Also required for hdf5.pxd
]
HDF4_ARGS = [
    '-Wall', '-fPIC', '-D_GNU_SOURCE', '-DHAVE_UUID', '-DHAVE_HDF4',
    '-ldfalt', '-lmfhdfalt', '-luuid',
]
# NOTE: For profile, add the following to MACROS:
#   ('CYTHON_NOGIL_TRACE', '1') # https://cython.readthedocs.io/en/latest/src/tutorial/profiling_tutorial.html

budget = Extension(
    name = 'budget',
    sources = ['budget.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'hdf5'],
    include_dirs = ['./utils', *HDF5_DIRS],
    extra_compile_args = ['-Wno-maybe-uninitialized'],
    extra_link_args = ['-fopenmp', '-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/']
)

core = Extension(
    name = 'core',
    sources = ['core.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'mfhdfalt'],
    include_dirs = HDF4_DIRS,
    extra_compile_args = HDF4_ARGS,
    extra_link_args = ['-fopenmp']
)

gpp = Extension(
    name = 'gpp',
    sources = ['gpp.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'hdf5'],
    include_dirs = ['./utils', *HDF5_DIRS],
    extra_compile_args = ['-Wno-maybe-uninitialized'],
    extra_link_args = ['-fopenmp', '-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/']
)

reco = Extension(
    name = 'reco',
    sources = ['reco.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'hdf5'],
    include_dirs = ['./utils', *HDF5_DIRS],
    extra_compile_args = ['-Wno-maybe-uninitialized'],
    extra_link_args = ['-fopenmp', '-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/']
)

resample = Extension(
    name = 'resample',
    sources = ['resample.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'hdf5'],
    include_dirs = ['./utils', *HDF5_DIRS],
    extra_compile_args = [
        '-DHAVE_HDF4', '-ldfalt', '-lhdf5', '-Wno-maybe-uninitialized'
    ],
    extra_link_args = ['-fopenmp', '-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/']
)

spinup = Extension(
    name = 'spinup',
    sources = ['spinup_9km.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'mfhdfalt'],
    include_dirs = ['./utils', *HDF4_DIRS],
    extra_compile_args = HDF4_ARGS,
    extra_link_args = ['-fopenmp']
)

setup(ext_modules = cythonize([core, resample, gpp, reco, budget, spinup]))
