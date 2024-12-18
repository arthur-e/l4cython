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


Ecosystem Respiration (RECO)
----------------------------

There are 1-km and 9-km versions of the RECO calculation. The 9-km simulation might look like:

```python
from l4cython.reco_9km import main

# Call the configuration file
main("L4Cython_RECO_M09_config.yaml")
```
