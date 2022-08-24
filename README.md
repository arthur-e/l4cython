L4C-Cython
===================

- [x] PFT map in sparse, binary format
- [ ] Routine to read-in BPLUT


Building
-------------------

```sh
cd l4cython
python setup.py build_ext --inplace
```


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
