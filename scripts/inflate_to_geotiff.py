'''
Inflates SOC restart files, then writes them to GeoTIFF format. Requires
that `pyl4c` is installed.
'''

import os
import warnings
import fire
import numpy as np
from l4cython.utils.mkgrid import inflate_file
from l4cython.utils.fixtures import NROW9KM, NCOL9KM, NROW1KM, NCOL1KM
from pyl4c.spatial import ease2_to_geotiff
from matplotlib import pyplot

def main(file_pattern, grid = 'M01', total = True):
    '''
    Parameters
    ----------
    file_pattern : str
        Should be string template, e.g., `L4Cython_Cnum%s_M09land.flt64` where
        `%s` stands in for the carbon pool number (0, 1, or 2)
    grid : str
        Either "M09" or "M01" (default)
    total : bool
        True (default) to also export the total SOC as a GeoTIFF file
    '''
    dtype = np.float32 if file_pattern.split('.')[-1] == 'flt32' else np.float64
    file_list = [file_pattern % i for i in range(0, 3)]
    for filename in file_list:
        # Skip files that were already inflated
        if os.path.exists(filename.replace(f'{grid}land', f'{grid}')):
            continue
        print(f'Inflating "{os.path.basename(filename)}"')
        inflate_file(filename, grid)
    file_list2 = [f.replace(f'{grid}land', f'{grid}') for f in file_list]
    array_list = [
        np.fromfile(filename, dtype) for filename in file_list2
    ]
    print('Reshaping array data...')
    stack = []
    for i in range(len(array_list)):
        arr = array_list[i]
        fname = file_list2[i]
        # Mask out NoData pixels
        arr[arr < 0] = np.nan
        if grid == 'M09':
            arr_ease2 = arr.reshape((NROW9KM, NCOL9KM))
        else:
            arr_ease2 = arr.reshape((NROW1KM, NCOL1KM))
        if total:
            stack.append(arr_ease2)
        ease2_to_geotiff(
            arr_ease2, '.'.join(fname.split('.')[:-1]) + '.tiff', grid = grid)
    if total:
        output_filename = '.'.join(file_pattern.split('.')[:-1]) + '_total.tiff'
        output_filename = output_filename.replace('%d', '')
        ease2_to_geotiff(
            np.stack(stack, axis = 0).sum(axis = 0),
            output_filename, grid = grid)


if __name__ == '__main__':
    fire.Fire(main)
