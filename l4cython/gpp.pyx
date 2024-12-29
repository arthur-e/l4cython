# cython: language_level=3

'''
'''

import cython
import datetime
import yaml
import numpy as np
from libc.stdio cimport FILE, fopen, fread, fclose, fwrite
from libc.math cimport fmax
from cython.parallel import prange
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from l4cython.constraints cimport is_valid, linear_constraint
from l4cython.utils cimport BPLUT, open_fid, to_numpy
from l4cython.utils.mkgrid cimport inflate
from l4cython.utils.mkgrid import write_inflated
from l4cython.utils.fixtures import READ, WRITE, DFNT_FLOAT32, NCOL9KM, NROW9KM, N_PFT, load_parameters_table
from l4cython.utils.fixtures import SPARSE_M09_N as PY_SPARSE_M09_N
from tqdm import tqdm

# EASE-Grid 2.0 params are repeated here to facilitate multiprocessing (they
#   can't be Python numbers)
cdef:
    FILE* fid
    BPLUT PARAMS
    int   FILL_VALUE = -9999
    int   M01_NESTED_IN_M09 = 9 * 9
    long  SPARSE_M09_N = PY_SPARSE_M09_N # Number of grid cells in sparse ("land") arrays
    long  SPARSE_M01_N = M01_NESTED_IN_M09 * SPARSE_M09_N
    unsigned char* PFT
PFT = <unsigned char*> PyMem_Malloc(sizeof(unsigned char) * SPARSE_M01_N)

# Python arrays that want heap allocations must be global; this one is reused
#   for any array that needs to be written to disk (using NumPy)
OUT_M01 = np.full((SPARSE_M01_N,), np.nan, dtype = np.float32)


@cython.boundscheck(False)
@cython.wraparound(False)
def main(config = None, verbose = True):
    '''
    Forward run...

    Parameters
    ----------
    config : str or dict
    verbose : bool
    '''
    cdef:
        float* smrz0
        float* smrz
    smrz0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    smrz  = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)

    # Read in configuration file, then load state data
    if config is None:
        config = '../data/L4Cython_GPP_M01_config.yaml'
    if isinstance(config, str) and verbose:
        print(f'Using config file: {config}')
    if isinstance(config, str):
        with open(config, 'r') as file:
            config = yaml.safe_load(file)

    params = load_parameters_table(config['BPLUT'].encode('UTF-8'))
    for p in range(1, N_PFT + 1):
        PARAMS.smrz0[p] = params['smrz0'][0][p]
        PARAMS.smrz1[p] = params['smrz1'][0][p]

    load_state(config) # Load global state variables

    PyMem_Free(smrz0)
    PyMem_Free(smrz)


def load_state(config):
    '''
    Populates global state variables with data.

    Parameters
    ----------
    config : dict
        The configuration data dictionary
    '''
    # Allocate space, read in 1-km PFT map
    n_bytes = sizeof(unsigned char)*SPARSE_M01_N
    fid = open_fid(config['data']['PFT_map'].encode('UTF-8'), READ)
    fread(PFT, sizeof(unsigned char), <size_t>n_bytes, fid)
    fclose(fid)


cdef void write_resampled(bytes output_filename, float* array_data, int inflated = 1):
    '''
    Resamples a 1-km array to 9-km, then writes the output to a file.

    Parameters
    ----------
    output_filename : bytes
    array_data : *float
    inflated : int
        1 if the output array should be inflated to a 2D global EASE-Grid 2.0
    '''
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
    # Write a flat (1D) file or inflate the file and then write
    if inflated == 0:
        data_resampled.tofile(output_filename.decode('UTF-8'))
    else:
        write_inflated(output_filename, data_resampled, grid = 'M09')
