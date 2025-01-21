# cython: language_level=3

'''
Assumptions:

- The fPAR dataset (an HDF5 file) has a field "fpar_M01" that contains the
    fPAR data and a field "fpar_qc_M01" that contains the QC flags.
'''

import cython
import datetime
import yaml
import numpy as np
import h5py
from libc.stdio cimport FILE, fopen, fread, fclose, fwrite
from libc.math cimport fmax
from cython.parallel import prange
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from bisect import bisect_right
from tempfile import NamedTemporaryFile
from l4cython.constraints cimport is_valid, linear_constraint
from l4cython.science cimport rescale_smrz, vapor_pressure_deficit
from l4cython.utils cimport BPLUT, open_fid, to_numpy
from l4cython.utils.mkgrid import write_numpy_inflated, write_numpy_deflated, deflate_file
from l4cython.utils.hdf5 cimport read_hdf5
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
# 8-day composite periods, as ordinal days of a 365-day year
PERIODS = np.arange(1, 365, 8)
PERIODS_DATES = [
    datetime.datetime.strptime('2005-%03d' % p, '%Y-%j') for p in PERIODS
]
PERIODS_DATES_LEAP = [
    datetime.datetime.strptime('2004-%03d' % p, '%Y-%j') for p in PERIODS
]


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
        Py_ssize_t i
        float* smrz0
        float* smrz
        float* tmean
        float* tmin
        float* qv2m
        float* ps
        float* tsurf
        float* ft
        float* vpd
        unsigned char* fpar
        unsigned char* fpar_qc
    smrz0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    smrz  = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    tmean  = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    tmin  = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    qv2m  = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    ps    = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    tsurf = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    ft    = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    vpd   = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    fpar  = <unsigned char*> PyMem_Malloc(sizeof(unsigned char) * SPARSE_M01_N)
    fpar_qc = <unsigned char*> PyMem_Malloc(sizeof(unsigned char) * SPARSE_M01_N)

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
        PARAMS.lue[p] = params['LUE'][0][p]
        PARAMS.smrz0[p] = params['smrz0'][0][p]
        PARAMS.smrz1[p] = params['smrz1'][0][p]
        PARAMS.vpd0[p] = params['vpd0'][0][p]
        PARAMS.vpd1[p] = params['vpd1'][0][p]
        PARAMS.tmin0[p] = params['tmin0'][0][p]
        PARAMS.tmin1[p] = params['tmin1'][0][p]
        PARAMS.ft0[p] = params['ft0'][0][p]
        PARAMS.ft1[p] = params['ft1'][0][p]

    load_state(config) # Load global state variables
    date_start = datetime.datetime.strptime(config['origin_date'], '%Y-%m-%d')
    num_steps = int(config['daily_steps'])

    # Begin forward time stepping
    date_fpar_ongoing = None
    for step in tqdm(range(num_steps)):
        date = date_start + datetime.timedelta(days = step)
        date_str = date.strftime('%Y%m%d')
        doy = int(date.strftime('%j'))

        # Read in soil moisture ("smrz") data
        fid = open_fid(
            (config['data']['drivers']['smrz'] % date_str).encode('UTF-8'), READ)
        fread(smrz0, sizeof(float), <size_t>sizeof(float)*SPARSE_M09_N, fid)
        fclose(fid)

        # Read in fPAR data; to do so, we need to first find the nearest
        #   8-day composite date
        p = int(date.strftime('%j'))
        # Find the date of the fPAR data associated with this 8-day span
        if date.year % 4 == 0:
            date_fpar = PERIODS_DATES_LEAP[bisect_right(PERIODS, p) - 1]
        else:
            date_fpar = PERIODS_DATES[bisect_right(PERIODS, p) - 1]

        # Only read-in the fPAR data (and deflate it) once, for every
        #   8-day period
        if date_fpar_ongoing is None or date_fpar_ongoing != date_fpar:
            # Load the corresponding fPAR data (as a NumPy array)
            fpar_filename = config['data']['drivers']['fpar'] % (
                str(date.year) + date_fpar.strftime('%m%d'))
            with h5py.File(fpar_filename, 'r') as hdf:
                fpar0 = hdf['fpar_M01'][:]
                fpar_qc0 = hdf['fpar_qc_M01'][:]
            out_fname = config['data']['scratch'] % ('fpar_M01', 'uint8')
            out_fname_qc = config['data']['scratch'] % ('fpar_qc_M01', 'uint8')
            fpar0.astype(np.uint8).tofile(out_fname)
            fpar_qc0.astype(np.uint8).tofile(out_fname_qc)
            deflate_file(out_fname)
            deflate_file(out_fname_qc)
            # Read in the deflated fPAR data
            fid = open_fid(
                (out_fname.replace('M01', 'M01land')).encode('UTF-8'), READ)
            fread(
                fpar, sizeof(unsigned char),
                <size_t>sizeof(unsigned char)*SPARSE_M01_N, fid)
            fclose(fid)
            # Read in the deflated fPAR QC flag data
            fid = open_fid(
                (out_fname_qc.replace('M01', 'M01land')).encode('UTF-8'), READ)
            fread(
                fpar_qc, sizeof(unsigned char),
                <size_t>sizeof(unsigned char)*SPARSE_M01_N, fid)
            fclose(fid)

            # Read in the remaining surface meteorlogical data
            # TODO See if we can use a single NamedTemporaryFile(),
            #   overwriting its contents without creating a new instance
            with h5py.File(
                    config['data']['drivers']['other'] % date_str, 'r') as hdf:
                # T2M_M09_MIN (tmin)
                tmp = NamedTemporaryFile()
                tmp_fname_bs = tmp.name.encode('UTF-8')
                write_numpy_deflated(tmp_fname_bs, hdf['T2M_M09_MIN'][:])
                fid = open_fid(tmp_fname_bs, READ)
                fread(
                    tmin, sizeof(float), <size_t>sizeof(float)*SPARSE_M09_N, fid)
                fclose(fid)
                # T2M_M09_AVG (tmean)
                tmp = NamedTemporaryFile()
                tmp_fname_bs = tmp.name.encode('UTF-8')
                write_numpy_deflated(tmp_fname_bs, hdf['T2M_M09_AVG'][:])
                fid = open_fid(tmp_fname_bs, READ)
                fread(
                    tmean, sizeof(float), <size_t>sizeof(float)*SPARSE_M09_N, fid)
                fclose(fid)
                # QV2M
                tmp = NamedTemporaryFile()
                tmp_fname_bs = tmp.name.encode('UTF-8')
                write_numpy_deflated(tmp_fname_bs, hdf['QV2M_M09_AVG'][:])
                fid = open_fid(tmp_fname_bs, READ)
                fread(
                    qv2m, sizeof(float), <size_t>sizeof(float)*SPARSE_M09_N, fid)
                fclose(fid)
                # PS
                tmp = NamedTemporaryFile()
                tmp_fname_bs = tmp.name.encode('UTF-8')
                write_numpy_deflated(tmp_fname_bs, hdf['SURFACE_PRESSURE'][:])
                fid = open_fid(tmp_fname_bs, READ)
                fread(
                    ps, sizeof(float), <size_t>sizeof(float)*SPARSE_M09_N, fid)
                fclose(fid)
            for i in prange(SPARSE_M09_N, nogil = True):
                vpd[i] = vapor_pressure_deficit(qv2m[i], ps[i], tmean[i])
            # NOTE: Alternatively, try reading with read_hdf5, then copying the
            #   array into an unsigned char* buffer
            # fname_bs = (config['data']['drivers']['other'] % date_str).encode('UTF-8')
            # read_hdf5(fname_bs, 'QV2M_M09_AVG', qv2m)

    PyMem_Free(smrz0)
    PyMem_Free(smrz)
    PyMem_Free(tmean)
    PyMem_Free(tmin)
    PyMem_Free(qv2m)
    PyMem_Free(ps)
    PyMem_Free(tsurf)
    PyMem_Free(ft)
    PyMem_Free(fpar)
    PyMem_Free(fpar_qc)
    PyMem_Free(vpd)


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


def mod15a2h_qc_fail(x):
    '''
    Returns pass/fail for QC flags based on the L4C fPAR QC protocol for the
    `FparLai_QC` band: Bad pixels have either `1` in the first bit ("Pixel not
    produced at all") or anything other than `00` ("clear") in bits 3-4.
    Output array is True wherever the array fails QC criteria. Compare to:

        np.vectorize(lambda v: v[0] == 1 or v[3:5] != '00')

    Parameters
    ----------
    x : numpy.ndarray
        Array where the last axis enumerates the unpacked bits
        (ones and zeros)

    Returns
    -------
    numpy.ndarray
        Boolean array with True wherever QC criteria are failed
    '''
    y = np.unpackbits(x[...,None], axis = 1)[...,-8:]
    # Emit 1 = FAIL if these two bits are not == "00"
    c1 = y[...,3:5].sum(axis = -1).astype(np.uint8)
    # Emit 1 = FAIL if 1st bit == 1 ("Pixel not produced at all")
    c2 = y[...,0]
    # Intermediate arrays are 1 = FAIL, 0 = PASS
    return (c1 + c2) > 0


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
        write_numpy_inflated(output_filename, data_resampled, grid = 'M09')
