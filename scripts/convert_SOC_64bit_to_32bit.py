'''
Converts soil organic carbon (SOC) spin-up files from 64-bit floating point
to 32-bit floating point.
'''

import warnings
import fire
import numpy as np

def main(file_pattern):
    '''
    Parameters
    ----------
    file_pattern : str
        Should be string template, e.g., `L4Cython_Cnum%s_M09land.flt64` where
        `%s` stands in for the carbon pool number (0, 1, or 2)
    '''
    file_list = [file_pattern % i for i in range(0, 3)]
    for filename in file_list:
        arr = np.fromfile(filename, np.float64)
        arr.astype(np.float32).tofile(filename.replace('.flt64', '.flt32'))


if __name__ == '__main__':
    fire.Fire(main)
