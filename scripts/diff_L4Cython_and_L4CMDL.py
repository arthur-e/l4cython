'''
Computes the difference between an L4Cython output sequence and an official
L4CMDL granule.
'''

import os
import warnings
import fire
import h5py
import numpy as np
import re
from l4cython.utils.mkgrid import inflate_file
from l4cython.utils.fixtures import NROW9KM, NCOL9KM, NROW1KM, NCOL1KM
from matplotlib import pyplot

FILE_RX = re.compile(r'L4Cython\_(?P<field>.*)\_(?P<date>\d{8})\_.*\.flt32')

def main(cython_granule, mdl_granule, grid = 'M09'):
    '''
    Parameters
    ----------
    cython_granule : str
        Should be string template, e.g., `L4Cython_Cnum%s_M09land.flt64` where
        `%s` stands in for the carbon pool number (0, 1, or 2)
    mdl_granule : str
    grid : str
        Either "M09" (default) or "M01"
    '''
    dtype = np.float32 if cython_granule.split('.')[-1] == 'flt32' else np.float64
    if f'{grid}land' in cython_granule:
        inflate_file(cython_granule, grid)
        filename = cython_granule.replace(f'{grid}land', f'{grid}')
    field, date = FILE_RX.match(os.path.basename(cython_granule)).groups()
    # Open the recent file
    recent = np.fromfile(cython_granule, dtype = dtype).reshape((1624, 3856))
    recent[recent < -900] = np.nan
    if field in ('Tmult', 'Wmult'):
        recent *= 100
    # Open the official V7 file
    with h5py.File(mdl_granule, 'r') as hdf:
        if field in ('Tmult', 'Wmult'):
            official = hdf[f'EC/{field.lower()}_mean'][:]
        else:
            official = hdf[f'{field}/{field.lower()}_mean'][:]
    official[official < -900] = np.nan
    # Statistics
    print(f'Official {field} on {date}:')
    print('-- ', np.nanpercentile(official, (0, 10, 50, 90, 100)).round(2))
    print(f'L4Cython {field} on {date}:')
    print('-- ', np.nanpercentile(recent, (0, 10, 50, 90, 100)).round(2))

    diff = official - recent
    if np.nanmax(np.abs(diff)) <= 1e-3 and np.nanmin(np.abs(diff)) <= 1e-3:
        print('Zero diff within tolerance of 1e-3')
    else:
        print(f'Global maximum (mean) difference: {np.nanmax(np.abs(diff)).round(3)} ({np.nanmean(np.abs(diff)).round(3)})')

    vlimit = max(np.nanmax(diff), np.nanmin(diff))
    pyplot.imshow(
        diff, interpolation = 'nearest', cmap = 'PRGn',
        vmin = -vlimit, vmax = vlimit)
    pyplot.colorbar()
    pyplot.title('Reference minus Prediction')
    pyplot.show()


if __name__ == '__main__':
    fire.Fire(main)
