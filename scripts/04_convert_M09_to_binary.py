'''
'''

import numpy as np
from pyl4c.spatial import as_array
from pyl4c.data.fixtures import EASE2_GRID_PARAMS

def main():
    for i in range(0, 3):
        print(f'Converting C{i}...')
        arr, _, _ = as_array(
            f'/ntsg_home/tcf_OL7000_C{i}_M09_0002089.tiff', band_axis = False)
        arr.astype(np.float32)\
            .tofile(f'/ntsg_home/tcf_OL7000_C{i}_M09_0002089.flt32')


if __name__ == '__main__':
    main()
