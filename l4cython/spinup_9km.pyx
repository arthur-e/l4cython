# cython: language_level=3

'''
Soil organic carbon (SOC) spin-up for SMAP Level 4 Carbon (L4C) model, based
on Version 6 state and parameters.

Takes about 130-150 seconds for the analytical spin-up.
'''

import cython
import datetime
import json
import numpy as np
from libc.math cimport fabs
from cython.parallel import prange
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from respiration cimport BPLUT, arrhenius, linear_constraint, to_numpy

# Number of grid cells in sparse ("land") arrays
DEF SPARSE_N = 1664040

# Additional Tsoil parameter (fixed for all PFTs)
cdef float TSOIL1 = 66.02 # deg K
cdef float TSOIL2 = 227.13 # deg K
# Additional SOC decay parameters (fixed for all PFTs)
cdef float KSTRUCT = 0.4 # Muliplier *against* base decay rate
cdef float KRECAL = 0.0093

# The PFT map
cdef unsigned char PFT[SPARSE_N]
cdef float ANNUAL_NPP[SPARSE_N]

# Python arrays that want heap allocations must be global; this one is reused
#   for any array that needs to be written to disk (using NumPy)
OUT_M09 = np.full((SPARSE_N,), np.nan, dtype = np.float32)

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
    '''
    cdef:
        int i
        float* soc0
        float* soc1
        float* soc2
    soc0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    soc1 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    soc2 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    # Read in configuration file, then load state data
    if config_file is None:
        config_file = '../data/L4Cython_spin-up_M09_config.json'
    with open(config_file) as file:
        config = json.load(file)
    load_state(config)
    # Step 1: Analytical spin-up
    # analytical_spinup(config, soc0, soc1, soc2)
    # OUT_M09 = to_numpy(soc0, SPARSE_N)
    # OUT_M09.tofile(
    #     '%s/L4Cython_Cana0_M09land.flt32' % config['model']['output_dir'])
    # OUT_M09 = to_numpy(soc1, SPARSE_N)
    # OUT_M09.tofile(
    #     '%s/L4Cython_Cana1_M09land.flt32' % config['model']['output_dir'])
    # OUT_M09 = to_numpy(soc2, SPARSE_N)
    # OUT_M09.tofile(
    #     '%s/L4Cython_Cana2_M09land.flt32' % config['model']['output_dir'])
    numerical_spinup(config, soc0, soc1, soc2)
    PyMem_Free(soc0)
    PyMem_Free(soc1)
    PyMem_Free(soc2)


cdef analytical_spinup(config, float* soc0, float* soc1, float* soc2):
    '''
    Analytical SOC spin-up: the initial soil C states are found by solving
    the differential equations that describe inputs, transfers, and decay
    of soil C.

    Parameters
    ----------
    config : dict
    float*: soc0
    float*: soc1
    float*: soc2
    '''
    cdef:
        Py_ssize_t i
        Py_ssize_t doy
        float* f_met
        float* f_str
        float* k0 # Decay rates of each pool
        float* k1
        float* k2
        float* k_mult
        float w_mult[SPARSE_N]
        float t_mult[SPARSE_N]
        float denom0, denom1
    f_met = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    f_str = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    k0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    k1 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    k2 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    k_mult = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    # For each day of the climatological year
    for doy in range(1, 366):
        jday = str(doy).zfill(3)
        # Convert to percentage units
        smsf = 100 * np.fromfile(
            config['data']['climatology']['smsf'] % jday, dtype = np.float32)
        tsoil = np.fromfile(
            config['data']['climatology']['tsoil'] % jday, dtype = np.float32)
        for i in range(0, SPARSE_N):
            pft = int(PFT[i])
            if pft not in (1, 2, 3, 4, 5, 6, 7, 8):
                continue
            # Just once for each pixel
            if doy == 1:
                # Initialize k_mult sum at zero; set base decay rates and
                #   allocation factors
                k_mult[i] = 0
                k0[i] = PARAMS.decay_rate[0][pft]
                k1[i] = PARAMS.decay_rate[1][pft]
                k2[i] = PARAMS.decay_rate[2][pft]
                f_met[i] = PARAMS.f_metabolic[pft]
                f_str[i] = PARAMS.f_structural[pft]
            # Compute and increment annual k_mult sum
            w_mult[i] = linear_constraint(
                smsf[i], PARAMS.smsf0[pft], PARAMS.smsf1[pft], 0)
            t_mult[i] = arrhenius(tsoil[i], PARAMS.tsoil[pft], TSOIL1, TSOIL2)
            k_mult[i] = k_mult[i] + (w_mult[i] * t_mult[i])
    # Outside of the daily loop, but for each pixel...
    for i in range(0, SPARSE_N):
        # Calculate analytical steady-state of SOC
        denom0 = (k_mult[i] * k0[i])
        if denom0 > 0:
            soc0[i] = (ANNUAL_NPP[i] * f_met[i]) / denom0
        else:
            soc0[i] = 0
        denom1 = (k_mult[i] * k1[i])
        if denom1 > 0:
            soc1[i] = (ANNUAL_NPP[i] * (1 - f_met[i])) / denom1
        else:
            soc1[i] = 0
        if k2[i] > 0:
            soc2[i] = (f_str[i] * k_mult[i] * soc1[i]) / k2[i]
        else:
            soc2[i] = 0
    PyMem_Free(f_met)
    PyMem_Free(f_str)
    PyMem_Free(k0)
    PyMem_Free(k1)
    PyMem_Free(k2)
    PyMem_Free(k_mult)


cdef numerical_spinup(config, float* soc0, float* soc1, float* soc2):
    '''
    Parameters
    ----------
    config : dict
    float*: soc0
    float*: soc1
    float*: soc2
    '''
    cdef:
        Py_ssize_t i
        Py_ssize_t doy
        float w_mult
        float t_mult
        float* k_mult
        float* k0
        float* k1
        float* k2
        float* f_met
        float* f_str
        float* diffs
    diffs = <float*> PyMem_Malloc(sizeof(float) * 3)
    k_mult = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    f_met = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    f_str = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    k0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    k1 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    k2 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    # Pre-allocate the decay rate and f_met, f_str arrays
    for i in range(SPARSE_N):
        pft = int(PFT[i])
        if pft not in (1, 2, 3, 4, 5, 6, 7, 8):
            continue
        f_met[i] = PARAMS.f_metabolic[pft]
        f_str[i] = PARAMS.f_structural[pft]
        k0[i] = PARAMS.decay_rate[0][pft]
        k1[i] = PARAMS.decay_rate[1][pft]
        k2[i] = PARAMS.decay_rate[2][pft]
    # Jones et al. (2017) write that goal is NEE tolerance <= 1 g C m-2 year-1
    tol = np.inf
    while np.abs(tol).max() > 1:
        # For each day of the climatological year
        print('Tolerance is...')
        for doy in range(1, 366):
            jday = str(doy).zfill(3)
            # Pre-compute k_mult
            smsf = 100 * np.fromfile( # Convert to percentage units
                config['data']['climatology']['smsf'] % jday, dtype = np.float32)
            tsoil = np.fromfile(
                config['data']['climatology']['tsoil'] % jday, dtype = np.float32)
            for i in range(SPARSE_N):
                pft = int(PFT[i])
                if pft not in (1, 2, 3, 4, 5, 6, 7, 8):
                    continue
                w_mult = linear_constraint(
                    smsf[i], PARAMS.smsf0[pft], PARAMS.smsf1[pft], 0)
                t_mult = arrhenius(tsoil[i], PARAMS.tsoil[pft], TSOIL1, TSOIL2)
                k_mult[i] = w_mult * t_mult
            for i in prange(SPARSE_N, nogil = True):
                numerical_step(
                    diffs, ANNUAL_NPP[i], k_mult[i], f_met[i], f_str[i],
                    k0[i], k1[i], k2[i], soc0[i], soc1[i], soc2[i])
                soc0[i] = soc0[i] + diffs[0]
                soc1[i] = soc1[i] + diffs[1]
                soc2[i] = soc2[i] + diffs[2]
            return
    PyMem_Free(diffs)
    PyMem_Free(k_mult)
    PyMem_Free(f_met)
    PyMem_Free(f_str)
    PyMem_Free(k0)
    PyMem_Free(k1)
    PyMem_Free(k2)


cdef void numerical_step(
        float* diffs, float litter, float k_mult, float f_met, float f_str,
        float k0, float k1, float k2, float c0, float c1, float c2) nogil:
    'A single daily soil decomposition step (for a single pixel)'
    cdef:
        float rh0
        float rh1
        float rh2
        float dc0
        float dc1
        float dc2
    rh0 = k_mult * k0 * c0
    rh1 = k_mult * k1 * c1
    rh2 = k_mult * k2 * c2
    # Calculate change in C pools (g C m-2 units)
    diffs[0] = (litter * f_met) - rh0
    diffs[1] = (litter * (1 - f_met)) - rh1
    diffs[2] = (f_str * rh1) - rh2


def load_state(config):
    '''
    Populates global state variables with data. Note that, compared to other
    respiration model variants, NPP here is the annual sum (i.e., not
    divided by 365.)

    Parameters
    ----------
    config : dict
        The configuration data dictionary
    '''
    PFT[:] = np.fromfile(config['data']['PFT_map'], np.uint8)
    ANNUAL_NPP[:] = np.fromfile(config['data']['NPP_annual_sum'], np.float32)
