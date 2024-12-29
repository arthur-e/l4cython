User Guide
===================


Program Modes
-------------------

The L4_C model consists of two sub-models, the ecosystem respiration (RECO) and gross primary productivity (GPP) models.

- The GPP model is fully vectorized and has no internal state; GPP is calculated as a function of the prevailing daily climate.
- The RECO model must be run forward with daily time steps, as it tracks changes to the internal soil organic carbon (SOC) state.

Net ecosystem exchange (NEE) can be calculated as part of the RECO forward run.

**In general, all program modes have a `main()` function that takes a single required argument: the configuration or `config` file. Check out config file templates here:**

```
l4cython/data
```


Required Data
-------------

**All program modes require the following datasets are provided (in the respective configuration file):**

- The `BPLUT` (Biome Properties Look-up Table)
- The Plant Functional Type Map, defined as `data/PFT_map` in the configuration file

A sample BPLUT is provided in the `data` folder for SMAP L4_C Version 7:

`data/L4C_Version7_BPLUT.csv`

The PFT map should be provided at 9-km or 1-km resolution, depending on the model.

**All required datasets should be provided in the configuration file as either:**

- A fully qualified file path to a specific file; or
- A fully qualified file path with a `%s` string formatting character, to be filled with the day of the simulation in `YYYYMMDD` (e.g., the surface soil moisture for day `YYYYMMDD`)


Ecosystem Respiration (RECO)
----------------------------

There are 1-km and 9-km versions of the RECO calculation. The 9-km simulation might look like:

```python
from l4cython.reco_9km import main

# Call the configuration file
main("L4Cython_RECO_M09_config.yaml")
```

**Required data:**

- The initial state of the three (3) soil organic carbon (SOC) pools: `data/SOC`
- The annual NPP sum (g C per meter squared per year): `data/NPP_annual_sum`
- The surface soil moisture (wetness, %): `data/drivers/smsf`
- The surface soil temperature (degrees K): `data/drivers/tsoil`
- The estimated GPP (g C per meter squared per day): `data/drivers/GPP`

**Optional data:**

- The litterfall schedule files, if `model/litterfall/scheduled` is set to `true`: `data/litterfall_schedule`
