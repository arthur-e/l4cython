'''
Extracts the 1-km SOC data in every pool from an operations SOC "restart" file
and writes to three output, 1D, binary flat-files.
'''

import numpy as np
import h5py
from l4cython.utils.mkgrid import write_numpy_deflated

def main(restart_file, output_soc_path_tpl, output_npp_path):
    '''
    Parameters
    ----------
    restart_file : str
    output_soc_path_tpl : str
    output_npp_path : str
    '''
    assert 'M01land' in output_soc_path_tpl,\
        'The output_soc_path_tpl must contain the string "M01land" for compatibility with mkgrid'
    assert '%d' in output_soc_path_tpl,\
        'The output_soc_path_tpl must contain a "%d" formatting string'
    with h5py.File(restart_file, 'r') as hdf:
        npp_field = list(filter(
            lambda name: not name.startswith('SOC'), list(hdf['SOC'].keys()))).pop()
        # Export the annual NPP sum
        print('Deflating annual NPP sum...')
        write_numpy_deflated(
            output_npp_path.encode('UTF-8'), hdf[f'SOC/{npp_field}'][:],
            grid = 'M01')

        # Export each SOC field
        soc_tpl_list = list(filter(
            lambda name: name.startswith('SOC'), list(hdf['SOC'].keys())))
        soc_tpl = soc_tpl_list.pop()
        # For all possible C%d...
        for i in range(3):
            soc_tpl = soc_tpl.replace(f'C{i}', 'C%d')
        for i in range(3):
            output_filename = output_soc_path_tpl % i
            if os.path.exists(output_filename):
                continue
            print('Deflating C%d...' % i)
            soc = hdf[f'SOC/{soc_tpl % i}'][:]
            write_numpy_deflated(output_filename.encode('UTF-8'), soc, grid = 'M01')


if __name__ == '__main__':
    import fire
    fire.Fire(main)
