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

- [ ] Single place for constants, e.g., `DFNT_FLOAT32`
- [ ] Move MODIS QA logic out of `gpp.pyx`, `budget.pyx`
- [x] MOVE `BPLUT` from `utils.__init__.pxd` into `core.pxd`
- [ ] MOVE all of `fixtures.py` into `core.pyx`
- https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html?highlight=packaging#distributing-cython-modules


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
