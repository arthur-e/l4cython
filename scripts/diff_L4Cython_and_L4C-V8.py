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
from matplotlib import pyplot

L4C_FILE_TPL = '/anx_lagr5/SMAP/L4_C/Version8/Vv8040/all_granules/SMAP_L4_C_mdl_{date}T000000_Vv8040_001.h5'
FILE_RX_BINARY = re.compile(r'L4Cython\_(?P<field>.*)\_(?P<date>\d{8})\_.*\.flt32')
FILE_RX_HDF5 = re.compile(r'L4Cython\_(?P<date>\d{8})\_.*\.h5')

def main(filename, grid = 'M09', plot = 'map', field = None, **plot_args):
    '''
    Parameters
    ----------
    file_pattern : str
        Should be string template, e.g., `L4Cython_Cnum%s_M09land.flt64` where
        `%s` stands in for the carbon pool number (0, 1, or 2)
    grid : str
        Either "M09" (default) or "M01"
    plot : str
        Either "map" (default) or "hist"
    field : str
        The field from the HDF5 granule to plot (optional)
    **plot_args
        Additional arguments to `matplotlib.pyplot()`
    '''
    dtype = np.float32 if '.flt32' in filename else np.float64
    if f'{grid}land' in filename:
        inflate_file(filename, grid)
        filename = filename.replace(f'{grid}land', f'{grid}')
    if '.h5' in filename:
        date = FILE_RX_HDF5.match(os.path.basename(filename)).groups()[0]
        with h5py.File(filename, 'r') as hdf:
            if field in ('Tmult', 'Wmult', 'Emult'):
                recent = hdf[f'EC/{field.lower()}_mean'][:]
            else:
                recent = hdf[f'{field}/{field.lower()}_mean'][:]
    else:
        field, date = FILE_RX_BINARY.match(os.path.basename(filename)).groups()
        # Open the recent RH file
        recent = np.fromfile(filename, dtype = dtype).reshape((1624, 3856))
    recent[recent < -9000] = np.nan
    if field in ('Tmult', 'Wmult', 'Emult'):
        recent *= 100
    # Open the official V7 file
    with h5py.File(L4C_FILE_TPL.format(date = date), 'r') as hdf:
        if field in ('Tmult', 'Wmult', 'Emult'):
            official = hdf[f'EC/{field.lower()}_mean'][:]
        else:
            official = hdf[f'{field}/{field.lower()}_mean'][:]
    official[official < -9000] = np.nan

    # Statistics
    print(f'Official V8 {field} on {date}:')
    print('-- ', np.nanpercentile(official, (0, 10, 50, 90, 100)).round(2))
    print(f'L4Cython {field} on {date}:')
    print('-- ', np.nanpercentile(recent, (0, 10, 50, 90, 100)).round(2))

    diff = official - recent
    if np.nanmax(np.abs(diff)) <= 1e-3 and np.nanmin(np.abs(diff)) <= 1e-3:
        print('Zero diff within tolerance of 1e-3')
    else:
        print(f'Global abs. maximum (mean) difference: {np.nanmax(np.abs(diff)).round(3)} ({np.nanmean(np.abs(diff)).round(3)})')

    if plot == 'map':
        vlimit = max(np.nanmax(diff), np.nanmin(diff))
        if 'vmin' not in plot_args.keys() and 'vmax' not in plot_args.keys():
            pyplot.imshow(
                diff, interpolation = 'nearest', cmap = 'PRGn',
                vmin = -vlimit, vmax = vlimit)
        else:
            pyplot.imshow(
                diff, interpolation = 'nearest', cmap = 'PRGn',
                **plot_args)
        pyplot.colorbar()
        pyplot.title('Reference minus Prediction')
        pyplot.show()
    elif plot == 'hist':
        idx = np.arange(0, official.size)
        idx = idx[~np.isnan(official.ravel())]
        idx = np.random.choice(idx, size = int(1e5))
        pyplot.hist(official.ravel()[idx], **plot_args)
        pyplot.title(f'Official V8 {field} on {date}')
        pyplot.show()
        pyplot.hist(official.ravel()[idx], **plot_args)
        pyplot.title(f'L4Cython {field} on {date}')
        pyplot.show()
        pyplot.ecdf(np.abs(diff.ravel()[idx]))
        pyplot.title(f'Difference: Official V8 minus L4Cython {field} on {date}')
        pyplot.show()

if __name__ == '__main__':
    fire.Fire(main)
