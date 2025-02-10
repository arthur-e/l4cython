'''
Extracts the 9-km mean GPP from an "L4CMDL" granule
and writes to output, 1D, binary flat-file.
'''

import numpy as np
import h5py
from l4cython.utils.mkgrid import write_numpy_deflated

def main(mdl_file, output_path):
    '''
    Parameters
    ----------
    mdl_file : str
    output_path : str
    '''
    assert 'M09land' in output_path,\
        'The output_path must contain the string "M09land" for compatibility with mkgrid'
    with h5py.File(mdl_file, 'r') as hdf:
        arr = hdf['GPP/gpp_mean'][:]
        arr[arr == -9999] = np.nan
        write_numpy_deflated(output_path.encode('UTF-8'), arr, grid = 'M09')


if __name__ == '__main__':
    import fire
    fire.Fire(main)
