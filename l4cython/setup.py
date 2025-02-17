# cython: language_level=3

from distutils.core import setup, Extension
from Cython.Build import cythonize

MACROS = [ # Avoids warning "Using deprecated NumPy API"
    ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'), # https://stackoverflow.com/questions/52749662/using-deprecated-numpy-api
]
# NOTE: For profile, add the following to MACROS:
#   ('CYTHON_NOGIL_TRACE', '1') # https://cython.readthedocs.io/en/latest/src/tutorial/profiling_tutorial.html

budget = Extension(
    name = 'budget',
    sources = ['budget.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt', 'hdf5'],
    include_dirs = [
        '/usr/include', '/usr/include/hdf', './utils',
        '/usr/include/hdf5/openmpi/', # Also required for hdf5.pxd
        '/usr/lib/x86_64-linux-gnu/openmpi/include/' # Also required for hdf5.pxd
    ],
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
    include_dirs = [
        '/usr/include', '/usr/include/hdf', './utils',
        '/usr/include/hdf5/openmpi/', # Also required for hdf5.pxd
        '/usr/lib/x86_64-linux-gnu/openmpi/include/' # Also required for hdf5.pxd
    ],
    extra_compile_args = [
        '-fopenmp', '-DHAVE_HDF4', '-ldfalt', '-lhdf5', '-Wno-maybe-uninitialized', '-g1'
    ],
    extra_link_args = ['-fopenmp', '-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/']
)

reco_9km = Extension(
    name = 'reco_9km',
    sources = ['reco_9km.pyx'],
    define_macros = MACROS,
    # -g1 : See: https://docs.cython.org/en/latest/src/userguide/faq.html#how-do-i-speed-up-the-c-compilation
    extra_compile_args = ['-g1']
)

reco = Extension(
    name = 'reco',
    sources = ['reco.pyx'],
    define_macros = MACROS,
    libraries = ['dfalt'],
    include_dirs = ['/usr/include', '/usr/include/hdf', './utils'],
    extra_compile_args = [
        '-fopenmp', '-DHAVE_HDF4', '-ldfalt', '-Wno-maybe-uninitialized', '-g1'
    ],
    extra_link_args = ['-fopenmp']
)

spinup = Extension(
    name = 'spinup',
    sources = ['spinup_9km.pyx'],
    define_macros = MACROS,
    extra_compile_args = ['-fopenmp', '-g1'],
    extra_link_args = ['-fopenmp']
)

setup(ext_modules = cythonize([budget, gpp, reco, reco_9km, spinup]))
