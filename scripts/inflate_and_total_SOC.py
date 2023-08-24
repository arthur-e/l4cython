'''
Inflates SOC restart files, stacks and totals them, then plots the data.
'''

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
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        print(np.nanpercentile(stack, (1, 10, 50, 90, 99), axis = 1).T.round(0))

    if grid == 'M09':
        total = stack.sum(axis = 0).reshape((NROW9KM, NCOL9KM))
    else:
        total = stack.sum(axis = 0).reshape((NROW1KM, NCOL1KM))
    pyplot.imshow(total, interpolation = 'nearest')
    pyplot.colorbar()
    pyplot.show()


if __name__ == '__main__':
    fire.Fire(main)
