# cython: language_level=3
# distutils: sources = ["l4cython/utils/src/spland.c", "l4cython/utils/src/uuta.c"]

'''
Generates annual SOC restart files (only). No output fluxes or daily SOC state
will be created. The intended use is for generating restart files ahead of a
forward run, which will allow for time-domain multiplexing, starting each job
at the beginning of a different year.
'''

import os
import cython
import datetime
import yaml
import numpy as np
import l4cython
from libc.stdlib cimport calloc, free
from libc.stdio cimport FILE, fread, fclose
from libc.math cimport fmax
from cython.parallel import prange
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from l4cython.core cimport BPLUT, FILL_VALUE, M01_NESTED_IN_M09, SPARSE_M09_N, SPARSE_M01_N, NCOL1KM, NROW1KM, NCOL9KM, NROW9KM, N_PFT, DFNT_UINT8, DFNT_FLOAT32
from l4cython.core import load_parameters_table
from l4cython.science cimport arrhenius, linear_constraint
from l4cython.utils.hdf5 cimport hid_t
from l4cython.utils.io cimport READ, open_fid, write_flat, read_flat, read_flat_short
from tqdm import tqdm

# EASE-Grid 2.0 params are repeated here to facilitate multiprocessing (they
#   can't be Python numbers)
cdef:
    FILE* fid
    BPLUT PARAMS
    # Additional Tsoil parameter (fixed for all PFTs)
    float TSOIL1 = 66.02 # deg K
    float TSOIL2 = 227.13 # deg K
    unsigned char* PFT
    float* LITTERFALL
    float* SOC0
    float* SOC1
    float* SOC2
PFT = <unsigned char*> PyMem_Malloc(sizeof(unsigned char) * SPARSE_M01_N)
LITTERFALL  = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
SOC0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
SOC1 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
SOC2 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)


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
        Py_ssize_t i, j, k, pft
        Py_ssize_t doy # Day of year, on [1,365]
        hid_t fid # For open HDF5 files
        int DEBUG, n_litter_days
        float litter # Amount of litterfall entering SOC pools
        float reco # Ecosystem respiration
        float k_mult

    # 9-km (M09) heap allocations, for RECO
    smsf  = <short int*> PyMem_Malloc(sizeof(short int) * SPARSE_M09_N)
    tsoil = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    # 1-km (M01) heap allocations, for GPP
    litter_rate = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    rh0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    rh1 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    rh2 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    rh_total = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    w_mult = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    t_mult = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    soc_total = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)

    # Read in configuration file, then load state data
    if config is None:
        config = os.path.join(
            os.path.dirname(l4cython.__file__), '../data/L4Cython_config.yaml')
    if isinstance(config, str) and verbose:
        print(f'Using config file: {config}')
    if isinstance(config, str):
        with open(config, 'r') as file:
            config = yaml.safe_load(file)
    load_state(config) # Load global state variables
    date_start = datetime.datetime.strptime(config['origin_date'], '%Y-%m-%d')
    num_steps = int(config['daily_steps'])
    DEBUG = 1 if config['debug'] else 0
    if DEBUG == 1:
        fpar_final = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)

    params = load_parameters_table(config['BPLUT'].encode('UTF-8'))
    for p in range(1, N_PFT + 1):
        PARAMS.smsf0[p] = params['smsf0'][0][p]
        PARAMS.smsf1[p] = params['smsf1'][0][p]
        PARAMS.tsoil[p] = params['tsoil'][0][p]
        PARAMS.cue[p] = params['CUE'][0][p]
        PARAMS.f_metabolic[p] = params['f_metabolic'][0][p]
        PARAMS.f_structural[p] = params['f_structural'][0][p]
        PARAMS.decay_rate[0][p] = params['decay_rate'][0][p]
        PARAMS.decay_rate[1][p] = params['decay_rate'][1][p]
        PARAMS.decay_rate[2][p] = params['decay_rate'][2][p]

    # Option to schedule the rate at which litterfall enters SOC pools; if no
    #   schedule is used, an equal daily fraction of available NPP allocated
    if config['model']['litterfall']['scheduled']:
        n_litter_days = config['model']['litterfall']['interval_days']
        n_litter_periods = int(np.ceil(365 / n_litter_days))
        periods = np.array([
            [i] * n_litter_days for i in range(1, n_litter_periods + 1)
        ]).ravel()
    else:
        # Allocate equal daily fraction; i.e., final rate is
        #   (litter_rate / n_litter_days) or 1/365
        n_litter_days = 365
        for i in range(0, SPARSE_M01_N):
            litter_rate[i] = 1

    #############################
    # Begin forward time stepping
    num_steps = int(config['daily_steps'])
    date_start = datetime.datetime.strptime(config['origin_date'], '%Y-%m-%d')
    for step in tqdm(range(num_steps)):
        date = date_start + datetime.timedelta(days = step)
        date_str = date.strftime('%Y%m%d')
        doy = int(date.strftime('%j'))
        year = int(date.year)

        # Read in soil moisture ("smsf" and "smrz") and soil temperature ("tsoil") data
        drivers = config['data']['drivers']
        read_flat_short((drivers['smsf'] % date_str).encode('UTF-8'), SPARSE_M09_N, smsf)
        read_flat((drivers['tsoil'] % date_str).encode('UTF-8'), SPARSE_M09_N, tsoil)

        # Option to schedule the rate at which litterfall enters SOC pools
        if config['model']['litterfall']['scheduled']:
            # Get the file covering the 8-day period in which this DOY falls
            filename = config['data']['litterfall_schedule']\
                % str(periods[doy-1]).zfill(2)
            read_flat(filename.encode('UTF-8'), SPARSE_M01_N, litter_rate)

        # Iterate over each 9-km pixel
        for i in prange(SPARSE_M09_N, nogil = True):
            # Iterate over each nested 1-km pixel
            for j in prange(M01_NESTED_IN_M09):
                # Hence, (i) indexes the 9-km pixel and k the 1-km pixel
                k = (M01_NESTED_IN_M09 * i) + j
                pft = PFT[k]
                # Make sure to fill output grids with the FILL_VALUE,
                #   otherwise they may contain zero (0) at invalid data
                w_mult[k] = FILL_VALUE
                t_mult[k] = FILL_VALUE
                rh_total[k] = FILL_VALUE
                soc_total[k] = FILL_VALUE
                if is_valid(pft, tsoil[i], LITTERFALL[k]) == 0:
                    continue # Skip invalid PFTs

                # Compute daily fraction of litterfall entering SOC pools
                litter = LITTERFALL[k] * (fmax(0, litter_rate[k]) / n_litter_days)
                # Compute daily RH based on moisture, temperature constraints;
                #   SMSF is in parts-per-thousand, convert to wetness (%)
                w_mult[k] = linear_constraint(
                    smsf[i] / 10.0, PARAMS.smsf0[pft], PARAMS.smsf1[pft], 0)
                t_mult[k] = arrhenius(tsoil[i], PARAMS.tsoil[pft], TSOIL1, TSOIL2)
                k_mult = w_mult[k] * t_mult[k]
                rh0[k] = k_mult * SOC0[k] * PARAMS.decay_rate[0][pft]
                rh1[k] = k_mult * SOC1[k] * PARAMS.decay_rate[1][pft]
                rh2[k] = k_mult * SOC2[k] * PARAMS.decay_rate[2][pft]
                # Calculate change in SOC pools; LITTERFALL[k] is daily litterfall
                SOC0[k] += (litter * PARAMS.f_metabolic[pft]) - rh0[k]
                SOC1[k] += (litter * (1 - PARAMS.f_metabolic[pft])) - rh1[k]
                SOC2[k] += (PARAMS.f_structural[pft] * rh1[k]) - rh2[k]
                # "the adjustment...to account for material transferred into the
                #   slow pool during humification" (Jones et al. 2017 TGARS, p.5)
                rh1[k] = rh1[k] * (1 - PARAMS.f_structural[pft])
                rh_total[k] = rh0[k] + rh1[k] + rh2[k]
                # Adjust pools, if needed, to guard against negative values
                rh_total[k] = fmax(rh_total[k], 0)
                SOC0[k] = fmax(SOC0[k], 0)
                SOC1[k] = fmax(SOC1[k], 0)
                SOC2[k] = fmax(SOC2[k], 0)
                soc_total[k] = SOC0[k] + SOC1[k] + SOC2[k]

        # Optionally create restart files for each C pool, at the beginning
        #   of a new year
        if config['model']['restart']['create_file']:
            if date.month == 12 and date.day == 31:
                filename = (config['model']['restart']['output_file']\
                    % (date_str, 0)).encode('UTF-8')
                write_flat(filename, SPARSE_M01_N, SOC0)
                filename = (config['model']['restart']['output_file']\
                    % (date_str, 1)).encode('UTF-8')
                write_flat(filename, SPARSE_M01_N, SOC1)
                filename = (config['model']['restart']['output_file']\
                    % (date_str, 2)).encode('UTF-8')
                write_flat(filename, SPARSE_M01_N, SOC2)

    PyMem_Free(PFT)
    PyMem_Free(LITTERFALL)
    PyMem_Free(SOC0)
    PyMem_Free(SOC1)
    PyMem_Free(SOC2)
    PyMem_Free(smsf)
    PyMem_Free(tsoil)
    PyMem_Free(litter_rate)
    PyMem_Free(rh0)
    PyMem_Free(rh1)
    PyMem_Free(rh2)
    PyMem_Free(rh_total)
    PyMem_Free(w_mult)
    PyMem_Free(t_mult)
    PyMem_Free(soc_total)


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
    # Allocate space for floating-point state datasets
    n_bytes = sizeof(float)*SPARSE_M01_N
    # Read in SOC datasets
    fid = open_fid(config['data']['SOC'][0].encode('UTF-8'), READ)
    fread(SOC0, sizeof(float), <size_t>n_bytes, fid)
    fclose(fid)
    fid = open_fid(config['data']['SOC'][1].encode('UTF-8'), READ)
    fread(SOC1, sizeof(float), <size_t>n_bytes, fid)
    fclose(fid)
    fid = open_fid(config['data']['SOC'][2].encode('UTF-8'), READ)
    fread(SOC2, sizeof(float), <size_t>n_bytes, fid)
    fclose(fid)
    # NOTE: Calculating litterfall as average daily NPP (constant fraction of
    #   the annual NPP sum)
    fid = open_fid(config['data']['NPP_annual_sum'].encode('UTF-8'), READ)
    fread(LITTERFALL, sizeof(float), <size_t>n_bytes, fid)
    fclose(fid)
    for i in range(SPARSE_M01_N):
        # Set any negative values (really just -9999) to zero
        SOC0[i] = fmax(0, SOC0[i])
        SOC1[i] = fmax(0, SOC1[i])
        SOC2[i] = fmax(0, SOC2[i])
        LITTERFALL[i] = fmax(0, LITTERFALL[i])


cdef inline char is_valid(char pft, float tsoil, float litter) nogil:
    '''
    Checks to see if a given pixel is valid, based on the PFT but also on
    select input data values. Modeled after `tcfModUtil_isInCell()` in the
    TCF code.

    Parameters
    ----------
    pft : char
        The Plant Functional Type (PFT)
    tsoil : float
        The daily mean soil temperature in the surface layer (deg K)
    litter : float
        The daily litterfall input

    Returns
    -------
    char
        A value of 0 indicates the pixel is invalid, otherwise returns 1
    '''
    cdef char valid = 1 # Assume it's a valid pixel
    if pft not in (1, 2, 3, 4, 5, 6, 7, 8):
        valid = 0
    elif tsoil <= 0:
        valid = 0
    elif litter <= 0:
        valid = 0
    return valid
