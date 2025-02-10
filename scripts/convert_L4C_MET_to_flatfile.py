'''
Extracts the 9-km surface meteorology from an "L4CMET" granule
and writes to output, 1D, binary flat-files.
'''

import numpy as np
import h5py
from l4cython.utils.mkgrid import write_numpy_deflated

FIELD_MAP = {
    'SM_SURFACE_WETNESS': 'smsf',
    'SOIL_TEMP_LAYER1': 'tsoil'
}

def main(met_file, output_path_tpl):
    '''
    Parameters
    ----------
    met_file : str
    output_path_tpl : str
    '''
    assert 'M09land' in output_path_tpl,\
        'The output_path_tpl must contain the string "M09land" for compatibility with mkgrid'
    assert '%s' in output_path_tpl,\
        'The output_path_tpl must contain a "%s" formatting string'
    with h5py.File(met_file, 'r') as hdf:
        for field, out_name in FIELD_MAP.items():
            output_filename = output_path_tpl % out_name
            arr = hdf[field][:]
            arr[arr < -9000] = np.nan
            if out_name in ('smsf', 'srmz'):
                # Convert to wetness
                arr *= 100
            write_numpy_deflated(output_filename.encode('UTF-8'), arr, grid = 'M09')


if __name__ == '__main__':
    import fire
    fire.Fire(main)
