[![DOI](https://zenodo.org/badge/1100770121.svg)](https://doi.org/10.5281/zenodo.17664242)


L4Cython (L4C in Cython)
========================

A Cython implementation of the Terrestrial Carbon Flux (TCF) model, which is
the basis for the Soil Moisture Active Passive (SMAP) Level 4 Carbon (L4C)
model. L4Cython is designed to reproduce the results of the operational NASA
SMAP L4C product (Jones et al. 2017; Endsley et al. 2020, 2026), but there will be
some differences due to the different implementations, usually on the order of
floating-point precision. L4Cython achieves the closest results with version
7.4.1 of the SMAP L4C operational code and Version 7 (Vv7040/Vv7042) of the
operational product. L4Cython is used to produce the SMAP L4C Nature Run.

For example, the GPP calculation in the `l4cython.gpp` module follows the
current (Version 7) official L4CMDL GPP calculation, where the 3-km field
`FT_STATE_UM_M03` from the NASA GMAO GEOS5 re-analysis is used to determine the
surface freeze-thaw state. However, in `l4cython.budget`, this field is not
used and the surface skin temperature is instead compared to the freezing point
of water.


Building and Testing
----------------------------

There are four steps involved:

1. Install Cython and other Python dependencies, e.g., with `pip`
2. Build and install `l4cython`
3. Install the `l4cython` Python module.
4. Update the file paths in `spland.h`.

**To install Cython and other Python dependencies,** from the project root directory:
```
pip install cython setuptools
```

**Build the `l4cython` module:**
```sh
# From the project root
make
```

**Then, install `l4cython` from the repository root:**

```sh
pip install -e .
```

**Finally, it is necessary to update the file `l4cython/utils/spland.h` to provide an absolute path to sparse land index files.** These files are located in the repository's `data` folder, but the file path will almost certainly be different on your file system so you need to update `spland.h` accordingly:

```c
/* spland.h */
#define   LAND_R_FILE  "/absolute/path/to/MCD12Q1_M09land_row.uint16"
#define   LAND_C_FILE  "/absolute/path/to/MCD12Q1_M09land_col.uint16"
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


References
----------------------------

- Endsley, K.A., J.S. Kimball, R.H. Reichle, J.V. Ardizzone, T. Kundig, T. Colligan, J. Heinke, E. Kato, J. Knauer, L. Ma, T. Nützel, Q. Sun, W. Yuan. 2026. ["A global, daily carbon budget for terrestrial ecosystems constrained by satellite observations of soil moisture: The SMAP Level 4 Carbon product at ten years."](https://doi.org/10.1029/2025JG009588) *Journal of Geophysical Research: Biogeosciences* 131(6).
- Endsley, K.A., J.S. Kimball, R.H. Reichle, J.D. Watts. 2020. ["Satellite monitoring of global surface soil organic carbon dynamics using the SMAP Level 4 Carbon product."](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2020JG006100) *Journal of Geophysical Research: Biogeosciences* 125(12).
