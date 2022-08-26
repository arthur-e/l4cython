L4C-Cython
===================

- [x] PFT map in sparse, binary format
- [ ] Routine to read-in BPLUT


Building and Troubleshooting
----------------------------

```sh
cd l4cython
python setup.py build_ext --inplace
```

The debugger `gdb` can be installed from source (needs Python 2 support, which isn't the default):

```
sudo apt install libgmp10 libgmp-dev python2.7-dev
./configure --with-python=/usr/bin/python2 --with-libgmp-prefix="/usr/lib/x86_64-linux-gnu"
make
sudo make install
```

`setup.py` needs to be amended:

```py
respiration = Extension(
    name = 'respiration',
    sources = ['test_1km.pyx'],
    define_macros = [ # Avoids warning "Using deprecated NumPy API"
        ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')
    ], # https://stackoverflow.com/questions/52749662/using-deprecated-numpy-api
    extra_compile_args = ['-g'],
    extra_link_args = ['-g']
)

setup(ext_modules = cythonize(respiration))
```

[Then see this resource for tips on debugging.](https://github.com/cython/cython/wiki/DebuggingTechniques)


Resources
-------------------

- [Cython for NumPy users](https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html#numpy-tutorial)
- [Guide to efficient NumPy implementation](https://cython.readthedocs.io/en/latest/src/tutorial/numpy.html)
- [Allocating memory for Typed Memoryviews](https://stackoverflow.com/questions/18462785/what-is-the-recommended-way-of-allocating-memory-for-a-typed-memory-view)


Implementation Details
----------------------

**Reading in large array data**

Memory views should be used by default, e.g.:

```py
cdef unsigned char PFT[SPARSE_N]
PFT[:] = np.fromfile('%s/SMAP_L4C_PFT_map_M09land.uint8' % ANC_DATA_DIR, np.uint8)
```

However, *despite the fact that memory views are supposed to automatically use heap allocations,* when the arrays are very large (e.g., 1-km data), attempting to build code that reads in these large arrays will result in memory overflow errors ("relocation truncated"). As such, manual (heap) memory allocation is required:

```py
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef unsigned char* PFT
PFT = <unsigned char*> PyMem_Malloc(sizeof(unsigned char) * SPARSE_M01_N)
for i, data in enumerate(np.fromfile('%s/SMAP_L4C_PFT_map_M01land.uint8' % ANC_DATA_DIR, np.uint8)):
    PFT[i] = data
```

Fortunately, the performance hit is small; for 9-km data:
```
# timeit mean time using memory views:
2.57 sec per loop

# timeit mean time using heap allocation
2.70 sec per loop
```


Concurrency
-------------------

OpenMP can be used for concurrency only if the contents of the inner-most loop (e.g., where `prange()` is used) contains only C code.

```
respration = Extension(
    ...
    extra_compile_args = ['-fopenmp'],
    extra_link_args = ['-fopenmp'])
```

[See this article for more information.](https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html#using-multiple-threads)
