'''
'''

import numpy as np
from pyl4c.spatial import as_array
from pyl4c.data.fixtures import EASE2_GRID_PARAMS
from pyl4c.lib.tcf import SparseArray

def main():
    for i in range(0, 3):
        print(f'Converting C{i}...')
        arr, _, _ = as_array(
            f'/ntsg_home/tcf_OL7000_C{i}_M09_0002089.tiff', band_axis = False)
        sparse = SparseArray(arr, grid = 'M09', dtype = np.float32)
        sparse.deflate()
        sparse.data.astype(np.float32)\
            .tofile(f'/anx_lagr3/arthur.endsley/SMAP_L4C/ancillary_data/tcf_OL7000_C{i}_M09land_0002089.flt32')

    # And now the annual NPP sum
    arr, _, _ = as_array(
        '/ntsg_home/tcf_OL7000_npp_sum_M09.tiff', band_axis = False)
    sparse = SparseArray(arr, grid = 'M09', dtype = np.float32)
    sparse.deflate()
    sparse.data.astype(np.float32)\
        .tofile('/anx_lagr3/arthur.endsley/SMAP_L4C/ancillary_data/tcf_OL7000_npp_sum_M09land.flt32')


if __name__ == '__main__':
    main()
