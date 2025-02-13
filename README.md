L4C-Cython
===================

A Cython implementation of the Terrestrial Carbon Flux (TCF) model, which is
the basis for the Soil Moisture Active Passive (SMAP) Level 4 Carbon (L4C) model.

- [ ] Single place for constants, e.g., `DFNT_FLOAT32`
- https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html?highlight=packaging#distributing-cython-modules
- *Getting to zero-diff with the operational L4CMDL product...*
  - Confirmed that the delivered SOC file (now as `/media/arthur.endsley/raid/TCF/tcf_V7_delivered_C*_M01land_0002089.flt32`) produces the closest results to the operational product.
  - RECO module achieves zero-diff when resampling to 9-km and comparing to `RH/rh_mean` or `NEE/nee_mean` (the latter, by using the GPP estimate from the L4CMDL granule).
  - GPP module does *not* achieve zero-diff in GPP, with 98% of differences less than $\pm 0.36 \text{g C m}^{-2} \text{day}^{-1}$. These appear to be due to unresolved differences in fPAR conditioning post-climatology fill. Differences in Emult are rare and resemble random noise.


Building and Testing
----------------------------

1. Install Cython and other Python dependencies.
2. Build the `utils` module.
3. Build the top-level `l4cython` modules.

**To install Cython and other Python dependencies,** from the project root directory:
```
pip install -e .
```

**Then, build the `utils` module:**
```sh
# From the project root
cd l4cython/utils
make
```

**Finally, build the top-level `l4cython` modules:**
```sh
# From the project root
cd l4cython
make
```

**If you have C dependency issues at compile time, note that some shared libraries have name variants on Ubuntu GNU/Linux and possibly other systems.** Symbolic linking is a simple fix.

```sh
sudo ln -s /usr/lib/libdfalt.so.0 /usr/lib/libdfalt.so
sudo ln -s /usr/lib/libmfhdfalt.so.0 /usr/lib/libmfhdfalt.so
```

**To test program modes, run `pytest` independently on each test suite** (there are issues with having `pytest` run them all at once:

```sh
pytest tests/test_utils.py
pytest tests/test_forward_run.py
pytest tests/test_forward_run_w_litterfall.py
pytest tests/test_spinup.py
```


Troubleshooting
----------------------------

If you have trouble locating dependencies on your system (e.g., `/usr/bin/ld: cannot find -ldf`), try using the `ld` utility.

Try calling `ld` in `--verbose` mode, e.g., to debug an issue with the `ldf` compiler flag:

```sh
$ ld -ldf --verbose
```


Debugging
----------------------------

**First, distinguish between typed memory views and manual heap allocation:**

```py
# Typed memory view
cdef:
    float rh0[SPARSE_N]

# Heap allocation
cdef
    float* rh0
rh0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
```

**Here are some tips:**

- **Don't mix typed memory views with manual heap allocation.** They can be used in the same Cython program, but you cannot combine them in a single computation step, it will lead to a segfault.

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


### Operations Not Permitted without the GIL

Some things to check if you get an error related to Cython code being interpreted as Python code inside a `prange()` loop, when the GIL is released:

- Are you using strictly C `inline` functions with a `nogil` annotation?
- Even if you are working with the return value of a valid C `inline` function, is the return value assigned to a C-defined (i.e., `cdef`) variable?


### Undefined Symbols in `*.pyx` Files

If you get an error, e.g.:

```sh
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ImportError: /usr/local/dev/l4cython/l4cython/gpp.cpython-310-x86_64-linux-gnu.so: undefined symbol: copyUUTA
make: *** [Makefile:17: test-gpp] Error 1
```

The above example is confusing because `copyUUTA` is indeed defined for the `mkgrid` extension when compiling the `utils` sub-module. We included the C source code at the top of `utils/mkgrid.pyx`:

```python
# distutils: sources = ["src/spland.c", "src/uuta.c"]
```

However, it's when we run code in the parent module that we see this error. It turns out that we still need to include this header in the parent module as well. The relative paths need to be changed, accordingly, at the top of `module.pyx`:

```python
# distutils: sources = ["utils/src/spland.c", "utils/src/uuta.c"]
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


On Spin-up to Steady-State Conditions
-------------------------------------

```
10 iterations is likely too few; from the tcf Vv6042 results, here are the
[1, 10, 50, 90, 99] percentiles of C0, C1, and C2:
    [ 31.,  67., 120., 244., 362.]
    [ 31.,  64., 119., 257., 398.]
    [ 563.,  939., 1708., 3312., 4542.]

5 iterations of tolerance = Delta NEE sum, with AD = [1, 1, 100]:
    [ 31.  66. 118. 202. 285.]
    [ 28.  63. 120. 246. 378.]
    [  641.  1272.  2228.  4469. 11755.]

10 iterations of tolerance = Delta NEE sum, with AD = [1, 1, 100]:
    [ 31.  66. 118. 202. 285.]
    [ 28.  63. 120. 246. 378.]
    [ 587. 1225. 2110. 3771. 5454.]

20 iterations of tolerance = Delta NEE sum, with AD = [1, 1, 100]:
    [ 31.  66. 118. 202. 285.]
    [ 28.  63. 120. 246. 378.]
    [ 583. 1224. 2107. 3760. 5470.]

10 iterations of tolerance = Delta NEE sum, with AD = [1, 1, 1]:
    [ 31.  66. 118. 202. 285.]
    [ 28.  63. 120. 246. 378.]
    [ 606. 1248. 2194. 3899. 5471.]

3 iterations (ended early) of tolerance = Delta NEE sum, with early quit
condition activated when "Pixels counted" went to zero (presumably because
they had all equilibrated); no AD:
    [ 31.  66. 118. 202. 285.]
    [ 28.  63. 120. 247. 378.]
    [ 606. 1248. 2194. 3900. 5472.]
```

A typical spin-up using the default configuration:

```
Beginning analytical spin-up...
100%|█████████████████████████████████████████████████████| 365/365 [00:09<00:00, 37.81it/s]
Beginning numerical spin-up...
100%|█████████████████████████████████████████████████████| 365/365 [38:20<00:00,  6.30s/it]
[1/10] Total tolerance is: 955233280.00
--- Pixels counted: 1354898
--- Mean tolerance is: 705.02
100%|█████████████████████████████████████████████████████| 365/365 [36:44<00:00,  6.04s/it]
[2/10] Total tolerance is: 1019234.62
--- Pixels counted: 1354867
--- Mean tolerance is: 0.75
100%|█████████████████████████████████████████████████████| 365/365 [15:49<00:00,  2.60s/it]
[3/10] Total tolerance is: 51874.89
--- Pixels counted: 582358
--- Mean tolerance is: 0.09
100%|█████████████████████████████████████████████████████| 365/365 [05:41<00:00,  1.07it/s]
[4/10] Total tolerance is: 10989.23
--- Pixels counted: 172253
--- Mean tolerance is: 0.06
100%|█████████████████████████████████████████████████████| 365/365 [01:31<00:00,  4.00it/s]
[5/10] Total tolerance is: -8208.76
--- Pixels counted: 37396
--- Mean tolerance is: -0.22
100%|█████████████████████████████████████████████████████| 365/365 [01:10<00:00,  5.14it/s]
[6/10] Total tolerance is: -45208.36
--- Pixels counted: 25337
--- Mean tolerance is: -1.78
100%|█████████████████████████████████████████████████████| 365/365 [00:36<00:00,  9.88it/s]
[7/10] Total tolerance is: -22261.44
--- Pixels counted: 10704
--- Mean tolerance is: -2.08
100%|█████████████████████████████████████████████████████| 365/365 [00:26<00:00, 13.82it/s]
[8/10] Total tolerance is: -19491.58
--- Pixels counted: 7200
--- Mean tolerance is: -2.71
100%|███████████████████████████████████████████████████████| 89/89 [00:04<00:00, 21.92it/s]
[9/10] Total tolerance is: 0.00
--- Pixels counted: 0
None
```
