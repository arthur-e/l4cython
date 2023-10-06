'''
Takes an M09land grid, inflates it, then scales it to M01 using nearest-
neighbor resampling, finally deflating to M01land.
'''

import os
import warnings
import fire
import numpy as np
from scipy.ndimage import zoom
from l4cython.utils.mkgrid import inflate_file, deflate_file
from l4cython.utils.fixtures import NROW9KM, NCOL9KM, NROW1KM, NCOL1KM
from matplotlib import pyplot

def main(file_path):
    '''
    Parameters
    ----------
    file_pattern : str
        Should be string template, e.g., `L4Cython_Cnum%s_M09land.flt64` where
        `%s` stands in for the carbon pool number (0, 1, or 2)
    grid : str
        Either "M09" (default) or "M01"
    '''
    dtype = np.float32 if file_path.split('.')[-1] == 'flt32' else np.float64
    assert 'M09land' in file_path, 'Does not appear to be an M09land file'
    inflate_file(file_path, 'M09')
    arr = np.fromfile(file_path.replace('M09land', 'M09'), dtype)
    scaled = zoom(
        arr.reshape((1624, 3856)), zoom = 9, grid_mode = True,
        mode = 'grid-constant')
    scaled.tofile(file_path.replace('M09land', 'M01'))
    deflate_file(file_path.replace('M09land', 'M01'), 'M01')


if __name__ == '__main__':
    fire.Fire(main)
