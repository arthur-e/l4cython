'''
Tests for the `spinup` module.
'''

import pytest
import yaml
import numpy as np
import l4cython
from pathlib import Path
from l4cython.utils.fixtures import NROW9KM, NCOL9KM, SPARSE_M09_N
from l4cython.spinup import main

PROJECT_ROOT = Path(l4cython.__path__[0]).parent
CONFIG_FILE = Path(PROJECT_ROOT, 'tests/data/L4Cython_spin-up_test_config.yaml')

def test_spinup_9km():
    ''
    stats_analytical = np.array([
        [ 11.6,  58. , 117. , 209.8, 658.7],
        [ 15.1,  57.5, 128.2, 268.4, 672.7],
        [ 316.8, 1292.9, 2432. , 4287.2, 8968.7],
    ])
    stats_numerical = np.array([
        [ 10.9,  59. , 125.1, 221.3, 682.2],
        [  16.5,  58.2, 134.3, 279.7, 702.3],
        [ 614.7, 1585., 3060., 9447.7, 153909.6],
    ])
    with open(CONFIG_FILE, 'r') as file:
        config = yaml.safe_load(file)
    output_dir = Path(config['model']['output_dir'])

    # This file is a good indicator of whether the file system is set up
    if not Path(config['data']['PFT_map']).exists():
        pytest.skip('Required input files not found')

    main(config, verbose = False)
    for pool in range(0, 3):
        filename = Path(output_dir, f'L4Cython_Cana{pool}_M09land.flt64')
        soc = np.fromfile(filename, np.float64)[0:1000]
        assert np.equal(
            stats_analytical[pool],
            np.percentile(soc, (0, 10, 50, 90, 100)).round(1)).all()
    for pool in range(0, 3):
        filename = Path(output_dir, f'L4Cython_Cnum{pool}_M09land_DOY365.flt64')
        soc = np.fromfile(filename, np.float64)[0:1000]
        assert np.equal(
            stats_numerical[pool],
            np.percentile(soc, (0, 10, 50, 90, 100)).round(1)).all()
