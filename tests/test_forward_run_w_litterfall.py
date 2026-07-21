'''
Tests forward model runs, including:

    l4cython.reco_9km.main()

This has to be done in a separate file, for some reason, because it would
otherwise cause a Segmentation Fault.
'''

import os
import yaml
import pytest
import numpy as np
import l4cython
from pathlib import Path
from l4cython.reco import main

PROJECT_ROOT = Path(l4cython.__path__[0]).parent
CONFIG_FILE = Path(PROJECT_ROOT, 'tests/data/L4Cython_RECO_test_config.yaml')

def test_forward_run_9km_with_litterfall_schedule():
    'Tests that 9-km run w/ litterfall schedule produces expected RH, NEE flux'
    with open(CONFIG_FILE, 'r') as file:
        config = yaml.safe_load(file)
    # Re-write the path to the BPLUT
    config['BPLUT'] = str(Path(PROJECT_ROOT, f'data/{config["BPLUT"]}'))
    config['model']['output_dir'] = str(Path(PROJECT_ROOT, 'tests/data'))
    config['model']['litterfall']['scheduled'] = True
    config['model']['litterfall']['interval_days'] = 8
    output_dir = Path(config['model']['output_dir'])
    data_dir = Path(PROJECT_ROOT, 'tests/data')

    main(config, verbose = False)
    rh_filename = Path(output_dir, f'L4Cython_RH_20150331_M09land.flt32')
    rh = np.fromfile(rh_filename, np.float32)
    rh[rh == -9999] = np.nan
    assert np.equal(
        np.nanpercentile(rh, (0, 10, 50, 90, 100)).round(2),
        np.array([0., 0.28, 4.1, 18.86, 45.74])).all()
    os.remove(rh_filename)
