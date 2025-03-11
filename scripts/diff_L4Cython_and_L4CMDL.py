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
from matplotlib import pyplot

FILE_RX = re.compile(r'L4Cython\_(?P<field>.*)\_(?P<date>\d{8})\_.*\.flt32')
HDF5_RX = re.compile(r'L4Cython\_(?P<date>\d{8})\_.*\.h5')

def main(cython_granule, mdl_granule, field = None, grid = 'M09'):
    '''
    Parameters
    ----------
    cython_granule : str
        The file path to an L4Cython output granule
    mdl_granule : str
        The file path to an L4CMDL output granule
    field : str or None
        Field to compare, if the L4Cython granule is an HDF5 file
    grid : str
        Either "M09" (default) or "M01"
    '''
    dtype = np.float32 if cython_granule.split('.')[-1] == 'flt32' else np.float64
    if f'{grid}land' in cython_granule and 'h5' not in cython_granule:
        inflate_file(cython_granule, grid)
        filename = cython_granule.replace(f'{grid}land', f'{grid}')
        field, date = FILE_RX.match(os.path.basename(cython_granule)).groups()
    elif 'h5' in cython_granule:
        date = HDF5_RX.match(os.path.basename(cython_granule)).groups()

    # Determine the HDF5 field name
    if field in ('Tmult', 'Wmult', 'Emult') or field.lower().startswith('f_'):
        field_name = f'EC/{field.lower()}_mean'
        field_name_official = 'EC/emult_mean'
    else:
        field_name = f'{field}/{field.lower()}_mean'
        field_name_official = field_name

    # Open the recent file
    if 'h5' in cython_granule:
        with h5py.File(cython_granule, 'r') as hdf:
            recent = hdf[field_name][:]
    else:
        recent = np.fromfile(
            cython_granule, dtype = dtype).reshape((1624, 3856))
    recent[recent < -900] = np.nan
    if field in ('Tmult', 'Wmult', 'Emult') or field.lower().startswith('f_'):
        recent *= 100

    # Open the official V7 file
    with h5py.File(mdl_granule, 'r') as hdf:
        official = hdf[field_name_official][:]
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
    pct_diff = ', '.join([
        '%.1f%%' % (100 * (diff[np.abs(diff) > thresh].size / diff.size) )
        for thresh in (1e-3, 1e-2, 1e-1)
    ])
    print(f'Percent differing by greater than (1e-3, 1e-2, 1e-1): {pct_diff}')

    vlimit = max(np.nanmax(diff), np.nanmin(diff))
    pyplot.imshow(
        diff, interpolation = 'nearest', cmap = 'PRGn',
        vmin = -vlimit, vmax = vlimit)
    pyplot.colorbar()
    pyplot.title('Reference minus Prediction')
    pyplot.show()


if __name__ == '__main__':
    fire.Fire(main)
