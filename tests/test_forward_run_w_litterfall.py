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
from l4cython.utils.fixtures import NROW9KM, NCOL9KM, SPARSE_M09_N
from l4cython.reco_9km import main

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
    config['data']['litterfall_schedule'] = 'L4Cython_litterfall_allocation_clim_2012-2021_M09land_P%s.flt32'
    for key in config['data'].keys():
        # It's a file path (as a string)
        if isinstance(config['data'][key], str):
            config['data'][key] = str(
                Path(PROJECT_ROOT, f'tests/data/{config["data"][key]}'))
        # It's a list of file paths
        elif hasattr(config['data'][key], 'append'):
            for i in range(len(config['data'][key])):
                config['data'][key][i] = str(
                    Path(PROJECT_ROOT, f'tests/data/{config["data"][key][i]}'))
        # It's a mapping of keys to file paths
        elif hasattr(config['data'][key], 'items'):
            for subkey in config['data'][key]:
                config['data'][key][subkey] = str(
                    Path(PROJECT_ROOT,
                        f'tests/data/{config["data"][key][subkey]}'))
    output_dir = Path(config['model']['output_dir'])
    data_dir = Path(PROJECT_ROOT, 'tests/data')

    # This file is a good indicator of whether the file system is set up
    if not Path(data_dir, config['data']['PFT_map']).exists():
        pytest.skip('Required input files not found')

    main(config, verbose = False)
    rh_filename = Path(output_dir, f'L4Cython_RH_20150331_M09land.flt32')
    nee_filename = Path(output_dir, f'L4Cython_NEE_20150331_M09land.flt32')
    rh = np.fromfile(rh_filename, np.float32)
    nee = np.fromfile(nee_filename, np.float32)
    assert np.equal(
        np.percentile(rh[0:1000], (0, 10, 50, 90, 100)).round(2),
        np.array([0.02, 0.30, 1.58, 4.02, 7.82])).all()
    assert np.equal(
        np.percentile(nee[0:1000], (0, 10, 50, 90, 100)).round(2),
        np.array([-4.6 , -1.2 ,  0.2 ,  1.27,  4.1 ])).all()
    os.remove(rh_filename)
    os.remove(nee_filename)
