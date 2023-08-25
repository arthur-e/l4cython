'''
TODO This is a placeholder for later, when mkgrid is renamed.
'''

import os
import numpy as np
from l4cython.utils.mkgrid import inflate_file

def main(filename, grid):
    inflate_file(filename, grid)
    infile = filename.replace('M09land', 'test')
    print(f'Reading from: {os.path.basename(infile)}')
    arr = np.fromfile(infile, np.float32)
    # import ipdb
    # ipdb.set_trace()#FIXME
    print(np.percentile(arr, (0, 10, 50, 90, 100)))


if __name__ == '__main__':
    import sys
    main(sys.argv[1], sys.argv[2])
