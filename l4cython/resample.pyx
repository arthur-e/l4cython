# cython: language_level=3

import os
import cython
import yaml
import numpy as np
import h5py
from tempfile import NamedTemporaryFile
from cython.parallel import prange
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from l4cython.core cimport FILL_VALUE, SPARSE_M01_N, SPARSE_M09_N, M01_NESTED_IN_M09
from l4cython.resample cimport write_fullres
from l4cython.utils.io cimport read_flat_short, to_numpy
from l4cython.utils.mkgrid import inflate_file, write_numpy_deflated
from l4cython.utils.hdf5 cimport hid_t, close_hdf5

PFT_VALID = (1,2,3,4,5,6,7,8) # Valid L4_C PFT classes

@cython.boundscheck(False)
@cython.wraparound(False)
def downscale(config, *granules, verbose = True):
    '''
    Downscales a SMAP L4CMDL granule to 1-km resolution using the 9-km pixel
    PFT-level mean values.

    Parameters
    ----------
    config : str or dict
        The path to the configuration file to use; or, a configuration that has
        already been read-in as a dictionary
    *granules : str
        Zero or more file paths, each for a SMAP SPL4CMDL granule
    verbose : bool
        True for verbose output
    '''
    cdef:
        hid_t fid # For open HDF5 files
        Py_ssize_t i, j, k, p9
    cdef char valid = 1 # Assume a pixel is valid
    cdef float* out_1km

    out_1km = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    in_9km = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    pft_map = <short int*> PyMem_Malloc(sizeof(short int) * SPARSE_M01_N)

    # Read in configuration file, then load the global state variables
    if isinstance(config, str) and verbose:
        print(f'Using config file: {config}')
    if isinstance(config, str):
        with open(config, 'r') as file:
            config = yaml.safe_load(file)
    if verbose:
        print('Finished reading config file')

    # Read in the land-cover map used for downscaling
    read_flat_short(
        config['land_cover_map'].encode('UTF-8'), SPARSE_M01_N, pft_map)
    if verbose:
        print('Finished reading land-cover file')

    # Downscale each granule
    output_dir = config['model']['output_dir']
    all_fields = config['fields']
    for each_granule in granules:
        fid = 0 # Reset the file identifier
        base_name = os.path.basename(each_granule).strip('.h5')
        if verbose:
            print(f'Working on "{os.path.basename(each_granule)}"...')
        with NamedTemporaryFile() as tmp:
            tmp_fname_bs = tmp.name.encode('UTF-8')
            for field in all_fields:
                with h5py.File(each_granule, 'r') as hdf:
                    # Only consider certain PFT classes in SMAP's model
                    for pft in config['classes']['l4c_allowed']:
                        field_bs = f'{field}/{field.lower()}_pft{pft}_mean'.encode('utf-8')
                        write_numpy_deflated(tmp.name, hdf[field_bs][:])
                        read_flat(tmp_fname_bs, SPARSE_M09_N, in_9km)

                        # Iterate over 9-km grid
                        for i in prange(SPARSE_M09_N, nogil = True):

                            # Iterate over each nested 1-km pixel
                            for j in range(M01_NESTED_IN_M09):
                                # Hence, (i) indexes the 9-km pixel and k the 1-km pixel
                                k = (M01_NESTED_IN_M09 * i) + j

                                # TODO How can this be done without hard-coding?
                                # Skip invalid pixels in the target land-cover map
                                if pft_map[k] not in (1,2,3,4,5,6):
                                    out_1km[k] = FILL_VALUE
                                    continue

                                # Skip invalid pixels from the input 9-km data;
                                #   i.e., no relevant data for this class
                                if in_9km[i] == FILL_VALUE:
                                    continue

                                # Otherwise, assign this (PFT mean) value to this
                                #   compatible 1-km pixel
                                out_1km[k] = in_9km[i]

                # For a given field
                base_name_bs = base_name.encode('UTF-8')
                field_bs = field.encode('UTF-8')
                # 1 is included for compatibility with write_resampled()
                fid = write_fullres(
                    config, out_1km, base_name_bs, field_bs, 1, fid)
                reset_array(out_1km)

        # Close the HDF5 file
        status = close_hdf5(fid)

    PyMem_Free(in_9km)
    PyMem_Free(out_1km)
    PyMem_Free(pft_map)


cdef inline float* reset_array(float* arr):
    '''
    Assigns the FILL_VALUE to all elements of an array, intended when arrays
    are re-used in a single session.

    Parameters
    ----------
    arr : float*
        Pointer to a 32-bit floating point array buffer

    Returns
    -------
    float*
    '''
    cdef Py_ssize_t i, j, k
    # Iterate over 9-km grid
    for i in prange(SPARSE_M09_N, nogil = True):
        # Iterate over each nested 1-km pixel
        for j in prange(M01_NESTED_IN_M09):
            # Hence, (i) indexes the 9-km pixel and k the 1-km pixel
            k = (M01_NESTED_IN_M09 * i) + j
            arr[k] = FILL_VALUE
    return arr
