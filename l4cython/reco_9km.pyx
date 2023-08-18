# cython: language_level=3

'''
SMAP Level 4 Carbon (L4C) heterotrophic respiration calculation, based on
Version 7 state and parameters, at 9-km spatial resolution.

Recent benchmarks on Gullveig (Intel Xeon 3.7 GHz): About 5s per data-day for
flat ("M09land") output.

Required daily driver data:

- Surface soil wetness ("SMSF"), in percentage units [0,100]
- Soil temperature, in degrees K
- Gross primary productivity (GPP), in [g C m-2 day-1]
'''

import cython
import datetime
import json
import numpy as np
from libc.stdio cimport FILE, fread, fclose
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from l4cython.respiration cimport BPLUT, arrhenius, linear_constraint
from l4cython.utils cimport open_fid
from l4cython.utils.mkgrid import write_inflated
from l4cython.utils.fixtures import READ
from tqdm import tqdm

# Number of grid cells in sparse ("land") arrays
DEF SPARSE_N = 1664040

# Additional Tsoil parameter (fixed for all PFTs)
cdef float TSOIL1 = 66.02 # deg K
cdef float TSOIL2 = 227.13 # deg K
# Additional SOC decay parameters (fixed for all PFTs)
cdef float KSTRUCT = 0.4 # Muliplier *against* base decay rate
cdef float KRECAL = 0.0093

cdef:
    FILE* fid
    unsigned char* PFT # The PFT map
    float* SOC0
    float* SOC1
    float* SOC2
    float* LITTERFALL
PFT = <unsigned char*> PyMem_Malloc(sizeof(unsigned char) * SPARSE_N)
SOC0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
SOC1 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
SOC2 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
LITTERFALL = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)

# L4_C BPLUT Version 7 (Vv7042, Vv7040, Nature Run v10)
# NOTE: BPLUT is initialized here because we *need* it to be a C struct and
#   1) It cannot be a C struct if it is imported from a *.pyx file (it gets
#   converted to a dict); 2) If imported as a Python dictionary and coerced
#   to a BPLUT struct, it's still (inexplicably) a dictionary; and 3) We can't
#   initalize the C struct's state if it is in a *.pxd file
cdef BPLUT PARAMS
# NOTE: Must have an (arbitrary) value in 0th position to avoid overflow of
#   indexing (as PFT=0 is not used and C starts counting at 0)
PARAMS.smsf0[:] = [0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
PARAMS.smsf1[:] = [0, 25.0, 94.0, 42.3, 35.8, 44.9, 52.9, 25.0, 25.0]
PARAMS.tsoil[:] = [0, 238.17, 422.77, 233.94, 246.48, 154.91, 366.14, 242.47, 265.06]
PARAMS.cue[:] = [0, 0.687, 0.469, 0.755, 0.799, 0.649, 0.572, 0.708, 0.705]
PARAMS.f_metabolic[:] = [0, 0.49, 0.71, 0.67, 0.67, 0.62, 0.76, 0.78, 0.78]
PARAMS.f_structural[:] = [0, 0.3, 0.3, 0.7, 0.3, 0.35, 0.55, 0.5, 0.8]
PARAMS.decay_rate[0] = [0, 0.020, 0.022, 0.030, 0.029, 0.012, 0.026, 0.018, 0.031]
for p in range(1, 9):
    PARAMS.decay_rate[1][p] = PARAMS.decay_rate[0][p] * KSTRUCT
    PARAMS.decay_rate[2][p] = PARAMS.decay_rate[0][p] * KRECAL


@cython.boundscheck(False)
@cython.wraparound(False)
def main(config_file = None):
    '''
    Forward run of the L4C soil decomposition and heterotrophic respiration
    algorithm. Starts on March 31, 2015 and continues for the specified
    number of time steps.
    '''
    cdef:
        Py_ssize_t i
        float* rh0
        float* rh1
        float* rh2
        float* gpp
        float* w_mult
        float* t_mult
    rh0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    rh1 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    rh2 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    gpp = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    w_mult = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    t_mult = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)

    # Read in configuration file, then load state data
    if config_file is None:
        config_file = '../data/L4Cython_RECO_M09_config.json'
    with open(config_file) as file:
        config = json.load(file)

    load_state(config) # Load PFT map, SOC state, etc.

    # We leave these as NumPy arrays because they're easier to write to disk
    rh_total = np.full((SPARSE_N,), np.nan, dtype = np.float32)
    nee = np.full((SPARSE_N,), np.nan, dtype = np.float32)
    date_start = datetime.datetime.strptime(config['origin_date'], '%Y-%m-%d')
    num_steps = int(config['daily_steps'])

    for step in tqdm(range(num_steps)):
        date = date_start + datetime.timedelta(days = step)
        date = date.strftime('%Y%m%d')
        smsf = np.fromfile(
            config['data']['drivers']['smsf'] % date, dtype = np.float32)
        tsoil = np.fromfile(
            config['data']['drivers']['tsoil'] % date, dtype = np.float32)
        # Read in the GPP data
        fid = open_fid((config['data']['drivers']['GPP'] % date).encode('UTF-8'), READ)
        fread(gpp, sizeof(float), <size_t>sizeof(float)*SPARSE_N, fid)
        fclose(fid)

        for i in range(0, SPARSE_N):
            pft = int(PFT[i])
            if pft not in (1, 2, 3, 4, 5, 6, 7, 8):
                continue
            w_mult[i] = linear_constraint(
                smsf[i], PARAMS.smsf0[pft], PARAMS.smsf1[pft], 0)
            t_mult[i] = arrhenius(tsoil[i], PARAMS.tsoil[pft], TSOIL1, TSOIL2)
            rh0[i] = w_mult[i] * t_mult[i] * SOC0[i] * PARAMS.decay_rate[0][pft]
            rh1[i] = w_mult[i] * t_mult[i] * SOC1[i] * PARAMS.decay_rate[1][pft]
            rh2[i] = w_mult[i] * t_mult[i] * SOC2[i] * PARAMS.decay_rate[2][pft]
            # "the adjustment...to account for material transferred into the
            #   slow pool during humification" (Jones et al. 2017 TGARS, p.5)
            rh1[i] = rh1[i] * (1 - PARAMS.f_structural[pft])
            rh_total[i] = rh0[i] + rh1[i] + rh2[i]
            # Calculate change in SOC pools
            SOC0[i] += (LITTERFALL[i] * PARAMS.f_metabolic[pft]) - rh0[i]
            SOC1[i] += (LITTERFALL[i] * (1 - PARAMS.f_metabolic[pft])) - rh1[i]
            SOC2[i] += (PARAMS.f_structural[pft] * rh1[i]) - rh2[i]
            # NEE is equivalent to RH - NPP; can be an expensive calculation
            if 'NEE' in config['model']['output_fields']:
                nee[i] = rh_total[i] - (gpp[i] * PARAMS.cue[pft])
        # Write datasets to disk
        rh_filename = '%s/L4Cython_RH_%s_M09.flt32' % (config['model']['output_dir'], date)
        nee_filename = '%s/L4Cython_NEE_%s_M09.flt32' % (config['model']['output_dir'], date)
        if config['model']['output_format'] == 'M09land':
            if 'RH' in config['model']['output_fields']:
                np.array(rh_total).astype(np.float32)\
                    .tofile(rh_filename.replace('M09', 'M09land'))
            if 'NEE' in config['model']['output_fields']:
                np.array(nee).astype(np.float32)\
                    .tofile(nee_filename.replace('M09', 'M09land'))
        else:
            if 'RH' in config['model']['output_fields']:
                write_inflated(rh_filename.encode('UTF-8'), rh_total)
            if 'NEE' in config['model']['output_fields']:
                write_inflated(nee_filename.encode('UTF-8'), nee)
        PyMem_Free(PFT)
        PyMem_Free(SOC0)
        PyMem_Free(SOC1)
        PyMem_Free(SOC2)
        PyMem_Free(LITTERFALL)
        PyMem_Free(rh0)
        PyMem_Free(rh1)
        PyMem_Free(rh2)
        PyMem_Free(w_mult)
        PyMem_Free(t_mult)


def load_state(config):
    '''
    Populates global state variables with data.

    Parameters
    ----------
    config : dict
        The configuration data dictionary
    '''
    # Allocate space, read in 1-km PFT map
    n_bytes = sizeof(unsigned char)*SPARSE_N
    fid = open_fid(config['data']['PFT_map'].encode('UTF-8'), READ)
    fread(PFT, sizeof(unsigned char), <size_t>n_bytes, fid)
    fclose(fid)
    # Allocate space for floating-point state datasets
    n_bytes = sizeof(float)*SPARSE_N
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
    for i in range(SPARSE_N):
        LITTERFALL[i] = LITTERFALL[i] / 365
