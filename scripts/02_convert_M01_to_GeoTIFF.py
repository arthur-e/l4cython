'''
'''

import numpy as np
from pyl4c.spatial import ease2_to_geotiff
from pyl4c.data.fixtures import EASE2_GRID_PARAMS

def main():
    for i in range(0, 3):
        print(f'Converting C{i}...')
        arr = np.fromfile(f'/ntsg_home/tcf_OL7000_C{i}_M01_0002089.flt32', np.float32)\
            .reshape(EASE2_GRID_PARAMS['M01']['shape'])
        arr[arr < 0] = -9999
        ease2_to_geotiff(
            arr, f'/ntsg_home/tcf_OL7000_C{i}_M01_0002089.tiff', grid = 'M01')


if __name__ == '__main__':
    main()
