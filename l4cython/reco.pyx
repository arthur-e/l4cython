# cython: language_level=3

'''
SMAP Level 4 Carbon (L4C) heterotrophic respiration calculation, based on
Version 6 state and parameters. The `main()` routine is optimized for model
execution but it may take several seconds to load the state data.

After the initial state data are loaded it takes about 40-60 seconds per
data day.

Required data:

- Surface soil wetness ("SMSF"), in proportion units [0,1]
- Soil temperature, in degrees K
'''

import cython
import datetime
import json
import numpy as np
from tqdm import tqdm
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from respiration cimport BPLUT, arrhenius, linear_constraint, to_numpy

DEF M01_NESTED_IN_M09 = 9 * 9
# Number of grid cells in sparse ("land") arrays
DEF SPARSE_M09_N = 1664040
DEF SPARSE_M01_N = M01_NESTED_IN_M09 * SPARSE_M09_N

# Additional Tsoil parameter (fixed for all PFTs)
cdef float TSOIL1 = 66.02 # deg K
cdef float TSOIL2 = 227.13 # deg K
# Additional SOC decay parameters (fixed for all PFTs)
cdef float KSTRUCT = 0.4 # Muliplier *against* base decay rate
cdef float KRECAL = 0.0093

# Python arrays that want heap allocations must be global; this one is reused
#   for any array that needs to be written to disk (using NumPy)
OUT_M01 = np.full((SPARSE_M01_N,), np.nan, dtype = np.float32)

# Allocate memory for PFT, SOC and litterfall (NPP) files
cdef:
    unsigned char* PFT
    float* SOC0
    float* SOC1
    float* SOC2
    float* NPP
PFT = <unsigned char*> PyMem_Malloc(sizeof(unsigned char) * SPARSE_M01_N)
SOC0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
SOC1 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
SOC2 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
NPP = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)

# L4_C BPLUT Version 6 (Vv6042, Vv6040, Nature Run v9.1)
# NOTE: Must have an (arbitrary) value in 0th position to avoid overflow of
#   indexing (as PFT=0 is not used and C starts counting at 0)
cdef BPLUT PARAMS
PARAMS.smsf0[:] = [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
PARAMS.smsf1[:] = [0, 25.0, 30.5, 39.8, 31.3, 44.9, 50.5, 25.0, 25.1]
PARAMS.tsoil[:] = [0, 266.05, 392.24, 233.94, 265.23, 240.71, 261.42, 253.98, 281.69]
PARAMS.cue[:] = [0, 0.687, 0.469, 0.755, 0.799, 0.649, 0.572, 0.708, 0.705]
PARAMS.f_metabolic[:] = [0, 0.49, 0.71, 0.67, 0.67, 0.62, 0.76, 0.78, 0.78]
PARAMS.f_structural[:] = [0, 0.3, 0.3, 0.7, 0.3, 0.35, 0.55, 0.5, 0.8]
PARAMS.decay_rate[0] = [0, 0.020, 0.022, 0.031, 0.028, 0.013, 0.022, 0.019, 0.031]
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
        Py_ssize_t j
        Py_ssize_t k
        float* rh0
        float* rh1
        float* rh2
        float* rh_total
        float* w_mult
        float* t_mult
        float k_mult
    rh0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    rh1 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    rh2 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    rh_total = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    w_mult = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    t_mult = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    # Read in configuration file, then load state data
    if config_file is None:
        config_file = '../data/L4Cython_RECO_M01_config.json'
    with open(config_file) as file:
        config = json.load(file)
    load_state(config)
    date_start = datetime.datetime.strptime(config['origin_date'], '%Y-%m-%d')
    num_steps = int(config['daily_steps'])
    for step in tqdm(range(num_steps)):
        date = date_start + datetime.timedelta(days = step)
        date = date.strftime('%Y%m%d')
        # Convert to percentage units
        smsf = 100 * np.fromfile(
            config['data']['drivers']['smsf'] % date, dtype = np.float32)
        tsoil = np.fromfile(
            config['data']['drivers']['tsoil'] % date, dtype = np.float32)
        # Iterate over each 9-km pixel
        for i in range(0, SPARSE_M09_N):
            # Iterate over each nested 1-km pixel
            for j in range(0, M01_NESTED_IN_M09):
                # Hence, (i) indexes the 9-km pixel and k the 1-km pixel
                k = (M01_NESTED_IN_M09 * i) + j
                pft = int(PFT[k])
                if pft not in (1, 2, 3, 4, 5, 6, 7, 8):
                    continue # Skip invalid PFTs
                w_mult[k] = linear_constraint(
                    smsf[i], PARAMS.smsf0[pft], PARAMS.smsf1[pft], 0)
                t_mult[k] = arrhenius(tsoil[i], PARAMS.tsoil[pft], TSOIL1, TSOIL2)
                k_mult = w_mult[k] * t_mult[k]
                rh0[k] = k_mult * SOC0[k] * PARAMS.decay_rate[0][pft]
                rh1[k] = k_mult * SOC1[k] * PARAMS.decay_rate[1][pft] * KSTRUCT
                rh2[k] = k_mult * SOC2[k] * PARAMS.decay_rate[2][pft] * KRECAL
                # "the adjustment...to account for material transferred into the
                #   slow pool during humification" (Jones et al. 2017 TGARS, p.5)
                rh1[k] = rh1[k] * (1 - PARAMS.f_structural[pft])
                rh_total[k] = rh0[k] + rh1[k] + rh2[k]
                # Calculate change in SOC pools; NPP[i] is daily litterfall
                SOC0[k] += (NPP[k] * PARAMS.f_metabolic[pft]) - rh0[k]
                SOC1[k] += (NPP[k] * (1 - PARAMS.f_metabolic[pft])) - rh1[k]
                SOC2[k] += (PARAMS.f_structural[pft] * rh1[k]) - rh2[k]
        OUT_M01 = to_numpy(rh_total, SPARSE_M01_N)
        OUT_M01.tofile(
            '%s/L4Cython_RH_%s_M01land.flt32' % (config['model']['output_dir'], date))
    PyMem_Free(PFT)
    PyMem_Free(SOC0)
    PyMem_Free(SOC1)
    PyMem_Free(SOC2)
    PyMem_Free(NPP)
    PyMem_Free(rh0)
    PyMem_Free(rh1)
    PyMem_Free(rh2)
    PyMem_Free(rh_total)
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
    # Read in SOC and NPP data; this has to be done by element, it seems
    for idx, data in enumerate(
            np.fromfile(config['data']['PFT_map'], np.uint8)):
        PFT[idx] = data
    for idx, data in enumerate(
            np.fromfile(config['data']['SOC'][0], np.float32)):
        SOC0[idx] = data
    for idx, data in enumerate(
            np.fromfile(config['data']['SOC'][1], np.float32)):
        SOC1[idx] = data
    for idx, data in enumerate(
            np.fromfile(config['data']['SOC'][2], np.float32)):
        SOC2[idx] = data
    # NOTE: Calculating litterfall as average daily NPP (constant fraction of
    #   the annual NPP sum)
    for idx, data in enumerate(
            np.fromfile(config['data']['NPP_annual_sum'], np.float32) / 365):
        NPP[idx] = data
