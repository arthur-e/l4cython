L4Cython (L4C in Cython)
========================

A Cython implementation of the Terrestrial Carbon Flux (TCF) model, which is
the basis for the Soil Moisture Active Passive (SMAP) Level 4 Carbon (L4C)
model. L4Cython is designed to reproduce the results of the operational NASA
SMAP L4C product (Jones et al. 2017; Endsley et al. 2020), but there will be
some differences due to the different implementations, usually on the order of
floating-point precision. L4Cython achieves the closest results with version
7.4.1 of the SMAP L4C operational code and Version 7 (Vv7040/Vv7042) of the
operational product.

- [ ] **The remaining difference between L4Cython GPP calculation and the
  current (Version 7) official L4CMDL GPP calculation is that the latter does
  not use the surface temperature from L4_SM but instead a field
  `FT_STATE_UM_M03` that depends on GMAO GEOS5.**
- [ ] **Address potential memory leak in `inflate()` and `deflate()` functions,**
  which allocate memory for a returned C array.
- [TODO: Changing static links to dynamic links](https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html?highlight=packaging#distributing-cython-modules)


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

**If you have C dependency issues at compile time, note that some shared libraries have name variants on Ubuntu GNU/Linux and possibly other systems.** Symbolic linking is a simple fix, though the details will vary across systems. Here's an example from Ubuntu Linux 22.04.

```sh
# For HDF4 shared libraries
sudo ln -s /usr/lib/libdfalt.so.0 /usr/lib/libdfalt.so
sudo ln -s /usr/lib/libmfhdfalt.so.0 /usr/lib/libmfhdfalt.so

# For HDF5 shared libraries
ln -s /usr/lib/x86_64-linux-gnu/libhdf5_openmpi.so.103 /usr/lib/x86_64-linux-gnu/libhdf5.so
ln -s /usr/lib/x86_64-linux-gnu/libhdf5_openmpi_hl.so.100 /usr/lib/x86_64-linux-gnu/libhdf5_openmpi_hl.so

# For libgctp; General Cartographic Transformation Package
sudo ln -s /usr/lib/x86_64-linux-gnu/libgctp.so /usr/lib/x86_64-linux-gnu/libGctp.so
```

**To test program modes, run `pytest` independently on each test suite** (there are issues with having `pytest` run them all at once:

```sh
pytest tests/test_utils.py
pytest tests/test_forward_run.py
pytest tests/test_forward_run_w_litterfall.py
pytest tests/test_spinup.py
```


Output Data
----------------------------

1-km binary files in geographic space (2D) will only be generated for soil-organic carbon spin-up. Otherwise, the file must be 9-km resolution or must an HDF5 output type. Additionally, land-format (1D, e.g, `M09land` or `M01land`) data will never be written to an HDF5 file.

| Module         | Output format | Output file type | Implemented?                 |
|:---------------|:--------------|:-----------------|:-----------------------------|
| `gpp.pyx`      | `M09`         | HDF5             | Yes, with `write_resampled()`|
| `gpp.pyx`      | `M09land`     | binary           | Yes, with `write_resampled()`|
| `gpp.pyx`      | `M01`         | HDF5             | No                           |
| `gpp.pyx`      | `M01land`     | binary           | Yes, with `to_numpy()`       |
| `reco.pyx`     | `M09`         | HDF5             | Yes, with `write_resampled()`|
| `reco.pyx`     | `M09land`     | binary           | Yes, with `write_resampled()`|
| `reco.pyx`     | `M01`         | HDF5             | No                           |
| `reco.pyx`     | `M01land`     | binary           | Yes, with `to_numpy()`       |
| `budget.pyx`   | `M09`         | HDF5             | Yes, with `write_resampled()`|
| `budget.pyx`   | `M09land`     | binary           | Yes, with `write_resampled()`|
| `budget.pyx`   | `M01`         | HDF5             | No                           |
| `budget.pyx`   | `M01land`     | binary           | Yes, with `to_numpy()`       |


Troubleshooting
----------------------------

If you have trouble locating dependencies on your system (e.g., `/usr/bin/ld: cannot find -ldf`), try using the `ld` utility.

Try calling `ld` in `--verbose` mode, e.g., to debug an issue with the `ldf` compiler flag:

```sh
$ ld -ldf --verbose
```
