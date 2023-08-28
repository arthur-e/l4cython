'''
Inflates SOC restart files, stacks and totals them, then plots the data.
'''

import os
import warnings
import fire
import numpy as np
from l4cython.utils.mkgrid import inflate_file
from l4cython.utils.fixtures import NROW9KM, NCOL9KM, NROW1KM, NCOL1KM
from matplotlib import pyplot

def main(file_pattern, grid = 'M09'):
    '''
    Parameters
    ----------
    file_pattern : str
        Should be string template, e.g., `L4Cython_Cnum%s_M09land.flt64` where
        `%s` stands in for the carbon pool number (0, 1, or 2)
    grid : str
        Either "M09" (default) or "M01"
    '''
    dtype = np.float32 if file_pattern.split('.')[-1] == 'flt32' else np.float64
    file_list = [file_pattern % i for i in range(0, 3)]
    for filename in file_list:
        inflate_file(filename, grid)
    file_list2 = [f.replace(f'{grid}land', f'{grid}') for f in file_list]
    stack = np.stack([
        np.fromfile(filename, dtype) for filename in file_list2
    ], axis = 0)
    stack[stack < 0] = np.nan

    # Also load the tolerance file
    tol_filename = os.path.join(os.path.dirname(file_pattern), f'L4Cython_numspin_tol_{grid}land.flt64')
    if os.path.exists(tol_filename):
        inflate_file(tol_filename, grid)
        tol = np.fromfile(tol_filename.replace(f'{grid}land', f'{grid}'), dtype)
        tol[tol <= -9999] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ptiles = np.nanpercentile(stack, (1, 10, 50, 90, 99), axis = 1).T.round(0)
        print(ptiles)

    if grid == 'M09':
        total = stack.sum(axis = 0).reshape((NROW9KM, NCOL9KM))
    else:
        total = stack.sum(axis = 0).reshape((NROW1KM, NCOL1KM))
    # Show up to the 99th percentile of the highest-storage SOC pool
    pyplot.imshow(total, interpolation = 'nearest', vmax = ptiles[2,4])
    pyplot.colorbar()
    pyplot.show()

    if os.path.exists(tol_filename):
        if grid == 'M09':
            tol = tol.reshape((NROW9KM, NCOL9KM))
        else:
            tol = tol.reshape((NROW1KM, NCOL1KM))
        p2, p98 = np.nanpercentile(tol, (2, 98))
        pyplot.imshow(tol, interpolation = 'nearest', vmin = p2, vmax = p98)
        pyplot.colorbar()
        pyplot.show()


if __name__ == '__main__':
    fire.Fire(main)
