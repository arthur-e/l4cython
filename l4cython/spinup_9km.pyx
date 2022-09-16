# cython: language_level=3

'''
Soil organic carbon (SOC) spin-up for SMAP Level 4 Carbon (L4C) model, based
on Version 6 state and parameters. Takes about 130-150 seconds for the
analytical spin-up.

After the first iteration (first climatological year) of the numerical
spin-up, the increments to the structural and recalcitrant pools (deltas)
may be so small in some areas that NaNs are emplaced. For a reference on
accelerated decomposition:

    https://doi.org/10.1016/j.ecolmodel.2005.04.008

Required data:

- Surface soil wetness ("SMSF"), in percentage units [0,100]
- Soil temperature, in degrees K

10 iterations is likely too few; from the tcf Vv6042 results, here are the
[1, 10, 50, 90, 99] percentiles of C0, C1, and C2:

    array([ 31.,  67., 120., 244., 362.])
    array([ 31.,  64., 119., 257., 398.])
    array([ 563.,  939., 1708., 3312., 4542.])

TODO I think litter rates need to be scaled as well. Rates might be scaled
differently for each pool...
'''

import cython
import datetime
import json
import numpy as np
from libc.math cimport isnan
from cython.parallel import prange
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from respiration cimport BPLUT, arrhenius, linear_constraint, to_numpy, to_numpy_double
from tqdm import tqdm

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
OUT_M09_DOUBLE = np.full((SPARSE_N,), np.nan, dtype = np.float64)

# L4_C BPLUT Version 6 (Vv6042, Vv6040, Nature Run v9.1)
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
        double* soc0
        double* soc1
        double* soc2
    soc0 = <double*> PyMem_Malloc(sizeof(double) * SPARSE_N)
    soc1 = <double*> PyMem_Malloc(sizeof(double) * SPARSE_N)
    soc2 = <double*> PyMem_Malloc(sizeof(double) * SPARSE_N)
    # Read in configuration file, then load state data
    if config_file is None:
        config_file = '../data/L4Cython_spin-up_M09_config.json'
    with open(config_file) as file:
        config = json.load(file)
    load_state(config)
    # Step 1: Analytical spin-up; note that soc0, soc1, soc2 are both
    #   inputs and outputs of the spin-up functions (K&R-style)
    analytical_spinup(config, soc0, soc1, soc2)
    OUT_M09_DOUBLE = to_numpy_double(soc0, SPARSE_N)
    OUT_M09_DOUBLE.tofile(
        '%s/L4Cython_Cana0_M09land.flt64' % config['model']['output_dir'])
    OUT_M09_DOUBLE = to_numpy_double(soc1, SPARSE_N)
    OUT_M09_DOUBLE.tofile(
        '%s/L4Cython_Cana1_M09land.flt64' % config['model']['output_dir'])
    OUT_M09_DOUBLE = to_numpy_double(soc2, SPARSE_N)
    OUT_M09_DOUBLE.tofile(
        '%s/L4Cython_Cana2_M09land.flt64' % config['model']['output_dir'])
    # Step 2: Numerical spin-up
    numerical_spinup(config, soc0, soc1, soc2)
    OUT_M09_DOUBLE = to_numpy_double(soc0, SPARSE_N)
    if config['model']['accelerated']:
        OUT_M09_DOUBLE *= config['model']['ad_rate']
    OUT_M09_DOUBLE.tofile(
        '%s/L4Cython_Cnum0_M09land.flt64' % config['model']['output_dir'])
    OUT_M09_DOUBLE = to_numpy_double(soc1, SPARSE_N)
    if config['model']['accelerated']:
        OUT_M09_DOUBLE *= config['model']['ad_rate']
    OUT_M09_DOUBLE.tofile(
        '%s/L4Cython_Cnum1_M09land.flt64' % config['model']['output_dir'])
    OUT_M09_DOUBLE = to_numpy_double(soc2, SPARSE_N)
    if config['model']['accelerated']:
        OUT_M09_DOUBLE *= config['model']['ad_rate']
    OUT_M09_DOUBLE.tofile(
        '%s/L4Cython_Cnum2_M09land.flt64' % config['model']['output_dir'])
    PyMem_Free(soc0)
    PyMem_Free(soc1)
    PyMem_Free(soc2)


cdef analytical_spinup(config, double* soc0, double* soc1, double* soc2):
    '''
    Analytical SOC spin-up: the initial soil C states are found by solving
    the differential equations that describe inputs, transfers, and decay
    of soil C.

    Parameters
    ----------
    config : dict
    double*: soc0
    double*: soc1
    double*: soc2
    '''
    cdef:
        Py_ssize_t i
        Py_ssize_t doy
        int pft
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
    print('Beginning analytical spin-up...')
    # For each day of the climatological year
    for doy in tqdm(range(1, 366)):
        jday = str(doy).zfill(3)
        # NOTE: It will ALWAYS be faster to read these all-at-once rather than
        #   to initialize them with heap allocations, then extract each
        #   element one-at-a-time (as in load_state())
        # NOTE: For compatibility with TCF, the SMSF data are already in
        #   percentage units, i.e., on [0,100]
        smsf = np.fromfile(
            config['data']['climatology']['smsf'] % jday, dtype = np.float32)
        tsoil = np.fromfile(
            config['data']['climatology']['tsoil'] % jday, dtype = np.float32)
        for i in range(SPARSE_N):
            pft = int(PFT[i])
            if pft == 0 or pft > 8:
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
    for i in prange(SPARSE_N, nogil = True):
        pft = int(PFT[i])
        if pft == 0 or pft > 8:
            continue
        # Calculate analytical steady-state of SOC
        denom0 = (k_mult[i] * k0[i])
        if denom0 > 0:
            soc0[i] = <double>(ANNUAL_NPP[i] * f_met[i]) / denom0
        else:
            soc0[i] = 0
        denom1 = (k_mult[i] * k1[i])
        if denom1 > 0:
            soc1[i] = <double>(ANNUAL_NPP[i] * (1 - f_met[i])) / denom1
        else:
            soc1[i] = 0
        # Guard against division by zero
        if k2[i] > 0:
            soc2[i] = <double>(f_str[i] * k1[i] * soc1[i]) / k2[i]
        else:
            soc2[i] = 0
    PyMem_Free(f_met)
    PyMem_Free(f_str)
    PyMem_Free(k0)
    PyMem_Free(k1)
    PyMem_Free(k2)
    PyMem_Free(k_mult)


cdef numerical_spinup(config, double* soc0, double* soc1, double* soc2):
    '''
    Parameters
    ----------
    config : dict
    double*: soc0
    double*: soc1
    double*: soc2
    '''
    cdef:
        Py_ssize_t i
        Py_ssize_t doy
        int iter
        int pft
        int success
        int tol_count # Number of pixels with tolerance, for calculating...
        float ad # Accelerated decomposition rate for litterfall
        float tol_mean # ...Overall mean tolerance
        float tol_sum # Sum of all tolerances
        float w_mult
        float t_mult
        float* k_mult
        float* k0
        float* k1
        float* k2
        float* f_met
        float* f_str
        double* delta # 3-element, recycling vector: diff. in each pool
        double* diffs # For each pixel, total diffs. over clim. year
        double* tolerance # Tolerance at each pixel
    k_mult = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    k0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    k1 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    k2 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    f_met = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    f_str = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    delta = <double*> PyMem_Malloc(sizeof(double) * 3)
    diffs = <double*> PyMem_Malloc(sizeof(double) * SPARSE_N)
    tolerance = <double*> PyMem_Malloc(sizeof(double) * SPARSE_N)
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
        # If accelerated decomposition is used
        if config['model']['accelerated']:
            k0[i] = k0[i] * config['model']['ad_rate']
            k1[i] = k1[i] * config['model']['ad_rate']
            k2[i] = k2[i] * config['model']['ad_rate']
        tolerance[i] = 0
    # For adjusting litterfall size if accelerated decomp. is used
    ad = 1.0
    if config['model']['accelerated']:
        ad = <float>(1 / config['model']['ad_rate'])
    print('Beginning numerical spin-up...')
    iter = 1
    success = 0
    while success != 1 and iter <= config['max_iter']:
        # Assume that we have fully equilibrated
        success = 1
        tol_sum = 0.0
        tol_count = 0
        # For each day of the climatological year
        for doy in tqdm(range(1, 366)):
            jday = str(doy).zfill(3)
            # Pre-compute k_mult
            # NOTE: For compatibility with TCF, the SMSF data are already in
            #   percentage units, i.e., on [0,100]
            smsf = np.fromfile(
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
            # Loop over pixels
            for i in prange(SPARSE_N, nogil = True):
                pft = int(PFT[i])
                if pft == 0 or pft > 8:
                    continue
                # Reset the annual Delta-NEE totals
                delta[0] = 0
                delta[1] = 0
                delta[2] = 0
                if doy == 1:
                    diffs[i] = 0
                # Compute one daily soil decomposition step for this pixel;
                #   note that litterfall is 1/365 of the annual NPP sum
                numerical_step(
                    delta, ad * (ANNUAL_NPP[i] / 365), k_mult[i], f_met[i],
                    f_str[i], k0[i], k1[i], k2[i], soc0[i], soc1[i], soc2[i])
                # Compute change in SOC storage as SOC + dSOC, i.e., "diffs"
                #   are the NEE/ change in SOC storage; it's unclear why, but
                #   we absolutely MUST test for NaNs here, not upstream
                if not isnan(delta[0]):
                    soc0[i] += delta[0]
                    diffs[i] += delta[0] # Add up the daily NEE/ dSOC changes
                if not isnan(delta[1]):
                    soc1[i] += delta[1]
                    diffs[i] += delta[1]
                if not isnan(delta[2]):
                    soc2[i] += delta[2]
                    diffs[i] += delta[2]
                # At end of year, calculate change in Delta-NEE relative to
                #   previous year
                if doy == 365:
                    if iter == 1:
                        # Overwrite the initial (large) arbitrary number
                        tolerance[i] = diffs[i]
                    else:
                        # i.e., Difference in Delta-NEE
                        tolerance[i] = tolerance[i] - diffs[i]
                    # Jones et al. (2017) write that goal is NEE
                    #   tolerance <= 1 g C m-2 year-1
                    if not isnan(tolerance[i]):
                        if tolerance[i] > 1:
                            success = 0
                        tol_count += 1
                        tol_sum += tolerance[i]
        print('Total tolerance is: %.2f' % tol_sum)
        print('Pixels counted: %d' % tol_count)
        if tol_count > 0:
            tol_mean = (tol_sum / tol_count)
            print('Mean tolerance is: %.2f' % tol_mean)
        iter = iter + 1
    OUT_M09_DOUBLE = to_numpy_double(tolerance, SPARSE_N)
    OUT_M09_DOUBLE.tofile(
        '%s/L4Cython_numspin_tol_M09land.flt64' % config['model']['output_dir'])
    PyMem_Free(k_mult)
    PyMem_Free(k0)
    PyMem_Free(k1)
    PyMem_Free(k2)
    PyMem_Free(f_met)
    PyMem_Free(f_str)
    PyMem_Free(delta)
    PyMem_Free(diffs)
    PyMem_Free(tolerance)


cdef void numerical_step(
        double* delta, float litter, float k_mult, float f_met, float f_str,
        float k0, float k1, float k2, double c0, double c1, double c2) nogil:
    'A single daily soil decomposition step (for a single pixel)'
    cdef:
        float rh0
        float rh1
        float rh2
    rh0 = k_mult * k0 * c0
    rh1 = k_mult * k1 * c1
    rh2 = k_mult * k2 * c2
    # Calculate change in C pools (g C m-2 units)
    if litter < 0:
        litter = 0
    else:
        delta[0] = <double>(litter * f_met) - rh0
        delta[1] = <double>(litter * (1 - f_met)) - rh1
        delta[2] = <double>(f_str * rh1) - rh2


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
