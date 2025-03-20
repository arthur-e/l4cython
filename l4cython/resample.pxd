# cython: language_level=3

import numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free
# NOTE: For some inexplicable reason, relative imports are needed here
from utils.mkgrid import write_numpy_inflated, write_numpy_deflated
from utils.hdf5 cimport hid_t, hsize_t, create_1d_space, create_2d_space, close_hdf5, open_hdf5, read_hdf5, write_hdf5_dataset, H5T_STD_U8LE, H5T_IEEE_F32LE
from utils.io cimport read_flat
from tempfile import NamedTemporaryFile

cdef extern from "utils/src/spland.h":
    int M01_NESTED_IN_M09
    int SPARSE_M09_N
    int SPARSE_M01_N
    int FILL_VALUE
    int NCOL1KM, NROW1KM, NCOL9KM, NROW9KM


cdef inline hid_t write_resampled(
        dict config, float* array_data, char* suffix, char* field,
        int inflated, hid_t file_id):
    '''
    Resamples a 1-km array to 9-km, then writes the output to a file.

    Parameters
    ----------
    config : dict
    array_data : float*
    field : char*
        Well-known name of the dataset to be written; optional for BINARY
        output
    suffix : char*
    inflated : int
        1 if the output array should be inflated to a 2D global EASE-Grid 2.0
    file_id : hid_t
        For HDF5 output, 0 if a new HDF5 file should be created; otherwise,
        pass the <hid_t> for an open HDF5 file

    Returns
    -------
    hid_t
        The file ID of the open HDF5 file, or 0 if no HDF5 file was used
    '''
    cdef float* data_inflated
    cdef hid_t fid
    cdef hid_t space_id
    cdef int grid_size
    if file_id > 0:
        fid = file_id
    else:
        fid = 0 # Nothing is open yet

    output_type = config['model']['output_type'].upper()
    output_dir = config['model']['output_dir']
    output_grid = config['model']['output_format']
    output_gzip = config['model']['compression']['level']
    assert output_type in ('HDF5', 'BINARY')
    assert field.decode('UTF-8') != '' or output_type == 'BINARY'
    _suffix = suffix.decode('UTF-8')
    _field = field.decode('UTF-8')

    # Specify the output resolution
    grid_size = 1000
    if 'M09' in output_grid:
        grid_size = 9000
    elif 'M03' in output_grid:
        grid_size = 3000

    # Resample the data from M01land to M09land
    data_resampled = FILL_VALUE * np.ones((SPARSE_M09_N,), np.float32)
    for i in range(0, SPARSE_M09_N):
        value = 0
        count = 0
        for j in range(0, M01_NESTED_IN_M09):
            k = (M01_NESTED_IN_M09 * i) + j
            if array_data[k] == FILL_VALUE:
                continue # Skip invalid PFTs
            value += array_data[k]
            count += 1
        if count == 0:
            continue
        value /= count
        data_resampled[i] = value

    # If writing a binary flat (1D) file, we're already pretty much done
    if output_type == 'BINARY':
        output_filename = (
            '%s/L4Cython_%s_%s.flt32' % (output_dir, _field, _suffix))
        if inflated == 1:
            write_numpy_inflated(
                output_filename, data_resampled, grid = 'M09')
        else:
            data_resampled.tofile(output_filename)
        return fid

    # Otherwise, it's HDF5 output; create the output HDF5 file
    output_filename = ('%s/L4Cython_%s.h5' % (output_dir, _suffix))\
        .encode('UTF-8')
    if fid == 0:
        fid = open_hdf5(output_filename)

    # Determine the output field name
    if _field in ('GPP', 'NPP'):
        _field = '%s/%s_mean' % (_field, _field.lower()) # e.g., 'GPP/gpp_mean'
    else:
        _field = 'EC/%s_mean' % _field.lower() # e.g., "EC/emult_mean"

    if inflated == 1:
        space_id = create_2d_space(NROW9KM, NCOL9KM)
        data_inflated = <float*> PyMem_Malloc(sizeof(float) * NCOL9KM * NROW9KM)
        # For HDF5 output, we'll first write the data to a temporary file
        tmp = NamedTemporaryFile()
        write_numpy_inflated(tmp.name, data_resampled, grid = 'M09')
        read_flat(tmp.name.encode('UTF-8'), NCOL9KM * NROW9KM, data_inflated)
        # Write the inflated data to a new HDF5 dataset
        write_hdf5_dataset(
            fid, _field.encode('UTF-8'), H5T_IEEE_F32LE, space_id,
            grid_size, output_gzip, data_inflated)
    else:
        raise NotImplementedError(
            'No support for writing deflated arrays to HDF5')

    PyMem_Free(data_inflated)
    return fid
