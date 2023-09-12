'''
Computes the difference between L4_C Version 7...
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

L4C_FILE = '/media/arthur.endsley/raid/SMAP/L4_C/Vv7042/SMAP_L4_C_mdl_20150331T000000_Vv7042_001.h5'
FILE_RX = re.compile(r'L4Cython\_(?P<field>.*)\_(?P<date>\d{8})\_M09\.flt32')

def main(filename, grid = 'M09'):
    '''
    Parameters
    ----------
    file_pattern : str
        Should be string template, e.g., `L4Cython_Cnum%s_M09land.flt64` where
        `%s` stands in for the carbon pool number (0, 1, or 2)
    grid : str
        Either "M09" (default) or "M01"
    '''
    dtype = np.float32 if filename.split('.')[-1] == 'flt32' else np.float64
    if 'M09land' in filename:
        inflate_file(filename, grid)
        filename = filename.replace(f'{grid}land', f'{grid}')
    field, date = FILE_RX.match(os.path.basename(filename)).groups()
    # Open the recent RH file
    recent = np.fromfile(filename, dtype = dtype).reshape((1624, 3856))
    recent[recent < 0] = np.nan
    if field in ('Tmult', 'Wmult'):
        recent *= 100
    # Open the official V7 file
    with h5py.File(L4C_FILE, 'r') as hdf:
        if field in ('Tmult', 'Wmult'):
            official = hdf[f'EC/{field.lower()}_mean'][:]
        else:
            official = hdf[f'{field}/{field.lower()}_mean'][:]
    official[official < 0] = np.nan
    # Statistics
    print(f'Official V7 {field} on 2015-03-31:')
    print('-- ', np.nanpercentile(official, (0, 10, 50, 90, 100)).round(2))
    print(f'L4Cython {field} on 2015-03-31:')
    print('-- ', np.nanpercentile(recent, (0, 10, 50, 90, 100)).round(2))

    diff = official - recent
    vlimit = max(np.nanmax(diff), np.nanmin(diff))
    pyplot.imshow(
        diff, interpolation = 'nearest', cmap = 'PRGn',
        vmin = -vlimit, vmax = vlimit)
    pyplot.colorbar()
    pyplot.title('Reference minus Prediction')
    pyplot.show()


if __name__ == '__main__':
    fire.Fire(main)
