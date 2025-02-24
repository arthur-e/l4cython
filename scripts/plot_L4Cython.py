'''
Plots the data from an L4Cython run.
'''

import os
import warnings
import fire
import glob
import h5py
import numpy as np
import re
from l4cython.utils.mkgrid import inflate_file
from l4cython.utils.fixtures import NROW9KM, NCOL9KM, NROW1KM, NCOL1KM
from matplotlib import pyplot

FILE_RX = re.compile(r'L4Cython\_(?P<field>.*)\_(?P<date>\d{8})\_.*\.flt32')
FILE_TPL = 'L4Cython_{field}_{date}_{grid}.flt32'

def main(file_path, field = None, grid = 'M09'):
    '''
    Parameters
    ----------
    file_path : str
        The directory where L4Cython outputs are stored
    field : str
        Optionally, only plot this given field
    grid : str
        Either "M09" (default) or "M01"
    '''
    if field is not None:
        file_glob = FILE_TPL.format(field = field, date = '*', grid = grid)
        file_list = glob.glob(os.path.join(file_path, file_glob))
        filename = file_list.pop()
        plot_field(filename, field, grid)
        return

    for field in ('GPP', 'Emult', 'Tmult', 'Wmult', 'RH', 'NEE'):
        file_glob = FILE_TPL.format(field = field, date = '*', grid = grid)
        file_list = glob.glob(os.path.join(file_path, file_glob))
        if len(file_list) == 0:
            print(f'WARNING: Could not find: "{file_glob}"')
            continue
        filename = file_list.pop()
        plot_field(filename, field, grid)


def plot_field(filename, field, grid = 'M09'):
    dtype = np.float32 if '.flt32' in filename else np.float64
    if f'{grid}land' in filename:
        inflate_file(filename, grid)
        filename = filename.replace(f'{grid}land', f'{grid}')
    field, date = FILE_RX.match(os.path.basename(filename)).groups()
    # Open the recent RH file
    recent = np.fromfile(filename, dtype = dtype).reshape((1624, 3856))
    recent[recent < -9000] = np.nan
    if field in ('Tmult', 'Wmult', 'Emult'):
        recent *= 100

    # Statistics
    print(f'L4Cython {field} on {date}:')
    print('-- ', np.nanpercentile(recent, (0, 10, 50, 90, 100)).round(2))

    vmin = np.nanpercentile(recent, 1)
    vmax = np.nanpercentile(recent, 99)
    pyplot.imshow(
        recent, interpolation = 'nearest', cmap = 'YlGnBu',
        vmin = vmin, vmax = vmax)
    pyplot.colorbar()
    pyplot.title(f'L4Cython: {field}')
    pyplot.show()



if __name__ == '__main__':
    fire.Fire(main)
