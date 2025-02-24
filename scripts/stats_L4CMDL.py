'''
Reports statistics for all fields of an L4CMDL granule.
'''

import os
import warnings
import fire
import glob
import h5py
import numpy as np
import re
from matplotlib import pyplot

def main(filename):
    '''
    Parameters
    ----------
    filename : str
    '''
    hdf = h5py.File(filename, 'r')
    for field in ('GPP', 'Emult', 'Tmult', 'Wmult', 'RH', 'NEE', 'SOC'):
        if 'mult' in field.lower():
            arr = hdf[f'EC/{field.lower()}_mean'][:]
        else:
            arr = hdf[f'{field}/{field.lower()}_mean'][:]
        arr[arr <= -9999] = np.nan
        print(field, np.nanpercentile(arr, (0, 10, 50, 90, 100)).round(3))
    hdf.close()


if __name__ == '__main__':
    fire.Fire(main)
