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

Accelerated decomposition rate coefficients of more than 100 tend to create
unstable pixels in the steady-state image.

Note that if the "data/NPP_annual_sum" configuration parameter is set to the
empty string, the model will use the daily GPP (converted to NPP) to determine
the daily NPP available for litterfall.

Recent benchmarks on Gullveig (Intel Xeon 3.7 GHz): 15 secs for analytical
spin-up, 25-30 min for the first climatological year in numerical spin-up. It
should get faster for each successive climatological year.

Required data:

- Surface soil wetness ("SMSF"), in percentage units [0,100]
- Soil temperature, in degrees K
'''

import cython
import datetime
import json
import numpy as np
from libc.math cimport isnan, fabs
from libc.stdio cimport FILE, fread, fclose
from cython.parallel import prange
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from l4cython.respiration cimport BPLUT, arrhenius, linear_constraint
from l4cython.utils cimport open_fid, to_numpy, to_numpy_double
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

# The PFT map
cdef:
    FILE* fid
    unsigned char* PFT
    float* AVAIL_NPP
PFT = <unsigned char*> PyMem_Malloc(sizeof(unsigned char) * SPARSE_N)
AVAIL_NPP = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)

# Python arrays that want heap allocations must be global; this one is reused
#   for any array that needs to be written to disk (using NumPy)
OUT_M09_DOUBLE = np.full((SPARSE_N,), np.nan, dtype = np.float64)

# L4_C BPLUT Version 7 (Vv7042, Vv7040, Nature Run v10)
cdef BPLUT PARAMS
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
    Soil organic carbon (SOC) spin-up; will write out a file for each soil
    pool after the analytical spin-up ("Cana") and after the numerical
    spin-up ("Cnum"), in addition to a final tolerance file.
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
    # If accelerated decomposition is used
    if config['model']['accelerated']:
        for i in range(SPARSE_N):
            soc0[i] = soc0[i] * config['model']['ad_rate'][0]
            soc1[i] = soc1[i] * config['model']['ad_rate'][1]
            soc2[i] = soc2[i] * config['model']['ad_rate'][2]
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
        OUT_M09_DOUBLE *= config['model']['ad_rate'][0]
    OUT_M09_DOUBLE.tofile(
        '%s/L4Cython_Cnum0_M09land.flt64' % config['model']['output_dir'])
    OUT_M09_DOUBLE = to_numpy_double(soc1, SPARSE_N)
    if config['model']['accelerated']:
        OUT_M09_DOUBLE *= config['model']['ad_rate'][1]
    OUT_M09_DOUBLE.tofile(
        '%s/L4Cython_Cnum1_M09land.flt64' % config['model']['output_dir'])
    OUT_M09_DOUBLE = to_numpy_double(soc2, SPARSE_N)
    if config['model']['accelerated']:
        OUT_M09_DOUBLE *= config['model']['ad_rate'][2]
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
        float* smsf
        float* tsoil
        float* f_met
        float* f_str
        float* param_smsf0
        float* param_smsf1
        float* param_tsoil
        float* k0 # Decay rates of each pool
        float* k1
        float* k2
        float* k_mult
        float* w_mult
        float* t_mult
        float denom0, denom1
    smsf = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    tsoil = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    f_met = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    f_str = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    param_smsf0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    param_smsf1 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    param_tsoil = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    k0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    k1 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    k2 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    k_mult = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    w_mult = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    t_mult = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    print('Beginning analytical spin-up...')
    # Analytical spin-up calculation has a closed form, so this outer loop,
    #   for each day of the climatological year, pre-computes annual sums
    for doy in tqdm(range(1, 366)):
        jday = str(doy).zfill(3)
        # NOTE: It will ALWAYS be faster to read these all-at-once rather than
        #   to initialize them with heap allocations, then extract each
        #   element one-at-a-time (as in load_state())
        # NOTE: For compatibility with TCF, the SMSF data are already in
        #   percentage units, i.e., on [0,100]
        fid = open_fid((config['data']['climatology']['smsf'] % jday).encode('UTF-8'), READ)
        fread(smsf, sizeof(float), <size_t>sizeof(float)*SPARSE_N, fid)
        fclose(fid)
        fid = open_fid((config['data']['climatology']['tsoil'] % jday).encode('UTF-8'), READ)
        fread(tsoil, sizeof(float), <size_t>sizeof(float)*SPARSE_N, fid)
        fclose(fid)
        # Vectorization: This loop initializes arrays that represent the
        #   parameter values for each pixel, to allow use in "nogil"
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
                if config['model']['accelerated']:
                    k0[i] = k0[i] * config['model']['ad_rate'][0]
                    k1[i] = k1[i] * config['model']['ad_rate'][1]
                    k2[i] = k2[i] * config['model']['ad_rate'][2]
                f_met[i] = PARAMS.f_metabolic[pft]
                f_str[i] = PARAMS.f_structural[pft]
                param_smsf0[i] = PARAMS.smsf0[pft]
                param_smsf1[i] = PARAMS.smsf1[pft]
                param_tsoil[i] = PARAMS.tsoil[pft]
        # Pre-compute the k_mult sum for each pixel
        for i in prange(SPARSE_N, nogil = True):
            pft = int(PFT[i])
            if pft == 0 or pft > 8:
                continue
            # Compute and increment annual k_mult sum
            w_mult[i] = linear_constraint(
                smsf[i], param_smsf0[i], param_smsf1[i], 0)
            t_mult[i] = arrhenius(tsoil[i], param_tsoil[i], TSOIL1, TSOIL2)
            k_mult[i] = k_mult[i] + (w_mult[i] * t_mult[i])
    # Outside of the daily loop, but for each pixel...
    for i in prange(SPARSE_N, nogil = True):
        pft = int(PFT[i])
        if pft == 0 or pft > 8:
            continue
        # Calculate analytical steady-state of SOC
        denom0 = (k_mult[i] * k0[i])
        if denom0 > 0:
            soc0[i] = <double>(365 * AVAIL_NPP[i] * f_met[i]) / denom0
        else:
            soc0[i] = 0
        denom1 = (k_mult[i] * k1[i])
        if denom1 > 0:
            soc1[i] = <double>(365 * AVAIL_NPP[i] * (1 - f_met[i])) / denom1
        else:
            soc1[i] = 0
        # Guard against division by zero
        if k2[i] > 0:
            soc2[i] = <double>(f_str[i] * k1[i] * soc1[i]) / k2[i]
        else:
            soc2[i] = 0
    PyMem_Free(smsf)
    PyMem_Free(tsoil)
    PyMem_Free(f_met)
    PyMem_Free(f_str)
    PyMem_Free(param_smsf0)
    PyMem_Free(param_smsf1)
    PyMem_Free(param_tsoil)
    PyMem_Free(k0)
    PyMem_Free(k1)
    PyMem_Free(k2)
    PyMem_Free(k_mult)
    PyMem_Free(w_mult)
    PyMem_Free(t_mult)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef numerical_spinup(config, double* soc0, double* soc1, double* soc2):
    '''
    Numerical SOC spin-up; the steady-state condition is defined by the inter-
    annual change in NEE (Delta-NEE) approaching zero. The near-zero condition
    or "tolerance" is defined as <1 g C m-2 year-1. Note that Jones et al.
    (2017) defined the steady state as when the annual NEE sum approaches
    zero. However, this condition is not guaranteed and, indeed, in the
    original L4C calibration code, it is actually Delta-NEE that is monitored
    for convergence. Note that an equivalent convergence diagnostic would be
    the change in SOC pools (Delta-SOC).

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
        int iter, pft, success
        int tol_count # Number of pixels with tolerance, for calculating...
        float tol_mean # ...Overall mean tolerance
        float tol_sum # Sum of all tolerances
        float w_mult, t_mult, k_mult
        float* smsf
        float* tsoil
        float* k0
        float* k1
        float* k2
        float* f_met
        float* f_str
        float* param_smsf0 # Vectorized parameter values
        float* param_smsf1
        float* param_tsoil
        float* gpp # Annual GPP sum
        float* ra_total # Annual autotrophic respiration (RA) sum
        float* rh_total # Annual heterotrophic respiration (RH) sum
        float* nee_sum # Annual NEE sum
        float* nee_last_year # Last year's annual NEE sum
        double* delta # 3-element, recycling vector: diff. in each pool
        double* tolerance # Tolerance at each pixel
    smsf = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    tsoil = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    k0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    k1 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    k2 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    f_met = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    f_str = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    param_smsf0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    param_smsf1 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    param_tsoil = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    gpp = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    ra_total = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    rh_total = <float*> PyMem_Malloc(sizeof(float) * 1)
    nee_sum = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    nee_last_year = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    delta = <double*> PyMem_Malloc(sizeof(double) * 3)
    tolerance = <double*> PyMem_Malloc(sizeof(double) * SPARSE_N)
    # Pre-allocate the decay rate, CUE, and f_met, f_str arrays
    for i in range(SPARSE_N):
        pft = int(PFT[i])
        if pft not in (1, 2, 3, 4, 5, 6, 7, 8):
            continue
        f_met[i] = PARAMS.f_metabolic[pft]
        f_str[i] = PARAMS.f_structural[pft]
        param_smsf0[i] = PARAMS.smsf0[pft]
        param_smsf1[i] = PARAMS.smsf1[pft]
        param_tsoil[i] = PARAMS.tsoil[pft]
        # Calculate annual GPP sum as (NPP / CUE)
        gpp[i] = (365 * AVAIL_NPP[i]) / PARAMS.cue[pft]
        ra_total[i] = gpp[i] - (365 * AVAIL_NPP[i])
        k0[i] = PARAMS.decay_rate[0][pft]
        k1[i] = PARAMS.decay_rate[1][pft]
        k2[i] = PARAMS.decay_rate[2][pft]
        # If accelerated decomposition is used
        if config['model']['accelerated']:
            k0[i] = k0[i] * config['model']['ad_rate'][0]
            k1[i] = k1[i] * config['model']['ad_rate'][1]
            k2[i] = k2[i] * config['model']['ad_rate'][2]
        tolerance[i] = 0
    print('Beginning numerical spin-up...')
    iter = 1
    success = 0
    tol_count = SPARSE_N # Initially, no pixels have met tolerance
    max_iter = config['max_iter']
    min_pixels = config['min_pixels'] # Minimum number of pixels remaining
    while success != 1 and iter <= max_iter and tol_count >= min_pixels:
        # Assume that we have fully equilibrated
        success = 1
        tol_sum = 0.0
        tol_count = 0
        # For each day of the climatological year
        for doy in tqdm(range(1, 366)):
            jday = str(doy).zfill(3)
            # NOTE: For compatibility with TCF, the SMSF data are already in
            #   percentage units, i.e., on [0,100]
            fid = open_fid((config['data']['climatology']['smsf'] % jday).encode('UTF-8'), READ)
            fread(smsf, sizeof(float), <size_t>sizeof(float)*SPARSE_N, fid)
            fclose(fid)
            fid = open_fid((config['data']['climatology']['tsoil'] % jday).encode('UTF-8'), READ)
            fread(tsoil, sizeof(float), <size_t>sizeof(float)*SPARSE_N, fid)
            fclose(fid)
            # Loop over pixels
            for i in prange(SPARSE_N, nogil = True):
                pft = int(PFT[i])
                if pft == 0 or pft > 8:
                    continue
                # Skip pixels that have already equilibrated
                if iter > 1 and not isnan(tolerance[i]):
                    if fabs(tolerance[i]) < 1:
                        continue
                # Reset the annual delta-SOC arrays
                delta[0] = 0
                delta[1] = 0
                delta[2] = 0
                # Compute one daily soil decomposition step for this pixel;
                #   note that litterfall is 1/365 of the annual NPP sum
                w_mult = linear_constraint(
                    smsf[i], param_smsf0[i], param_smsf1[i], 0)
                t_mult = arrhenius(tsoil[i], param_tsoil[i], TSOIL1, TSOIL2)
                k_mult = w_mult * t_mult
                numerical_step(
                    delta, rh_total, AVAIL_NPP[i], k_mult, f_met[i],
                    f_str[i], k0[i], k1[i], k2[i], soc0[i], soc1[i], soc2[i])
                # Compute change in SOC storage as SOC + dSOC, it's unclear
                #   why, but we absolutely MUST test for NaNs here, not upstream
                if not isnan(delta[0]):
                    soc0[i] += delta[0]
                if not isnan(delta[1]):
                    soc1[i] += delta[1]
                if not isnan(delta[2]):
                    soc2[i] += delta[2]
                # At end of year, calculate change in Delta-NEE relative to
                #   previous year
                if doy == 365:
                    # i.e., tolerance is the change in the Annual NEE sum:
                    #   NEE = (RA + RH) - GPP
                    #   DeltaNEE = NEE(t) - NEE(t-1)
                    nee_sum[i] = (ra_total[i] + rh_total[0]) - gpp[i]
                    # Jones et al. (2017) write that goal is NEE
                    #   tolerance <= 1 g C m-2 year-1
                    if iter > 0:
                        tolerance[i] = nee_last_year[i] - nee_sum[i]
                    else:
                        tolerance[i] = nee_sum[i]
                    nee_last_year[i] = nee_sum[i]
                    if not isnan(tolerance[i]):
                        if fabs(tolerance[i]) > 1:
                            success = 0
                        tol_count += 1
                        tol_sum += tolerance[i]
        print('[%d/%d] Total tolerance is: %.2f' % (iter, max_iter, tol_sum))
        print('--- Pixels counted: %d' % tol_count)
        if tol_count > 0:
            tol_mean = (tol_sum / tol_count)
            print('--- Mean tolerance is: %.2f' % tol_mean)
        else:
            print('QUIT early due to zero pixel count in tolerance calculation')
            break # Quit if we're not counting valid pixels anymore
        # Increment; also a counter for the number of climatological years
        iter = iter + 1
    OUT_M09_DOUBLE = to_numpy_double(tolerance, SPARSE_N)
    OUT_M09_DOUBLE.tofile(
        '%s/L4Cython_numspin_tol_M09land.flt64' % config['model']['output_dir'])
    PyMem_Free(smsf)
    PyMem_Free(tsoil)
    PyMem_Free(k0)
    PyMem_Free(k1)
    PyMem_Free(k2)
    PyMem_Free(f_met)
    PyMem_Free(f_str)
    PyMem_Free(param_smsf0)
    PyMem_Free(param_smsf1)
    PyMem_Free(param_tsoil)
    PyMem_Free(gpp)
    PyMem_Free(ra_total)
    PyMem_Free(rh_total)
    PyMem_Free(nee_sum)
    PyMem_Free(nee_last_year)
    PyMem_Free(delta)
    PyMem_Free(tolerance)


cdef void numerical_step(
        double* delta, float* rh_total, float litter, float k_mult,
        float f_met, float f_str, float k0, float k1, float k2,
        double c0, double c1, double c2) nogil:
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
    # Adjust structural RH pool for material transferred to recalcitrant
    rh1 = rh2 * (1 - f_str)
    rh_total[0] = rh0 + rh1 + rh2


def load_state(config):
    '''
    Populates global state variables with data.

    Parameters
    ----------
    config : dict
        The configuration data dictionary
    '''
    n_bytes = sizeof(unsigned char)*SPARSE_N
    fid = open_fid(config['data']['PFT_map'].encode('UTF-8'), READ)
    fread(PFT, sizeof(unsigned char), <size_t>n_bytes, fid)
    fclose(fid)
    if config['data']['NPP_annual_sum'] != '':
        fid = open_fid(config['data']['NPP_annual_sum'].encode('UTF-8'), READ)
        fread(AVAIL_NPP, sizeof(float), <size_t>n_bytes, fid)
        fclose(fid)
        for i in range(SPARSE_N):
            # Available NPP is 1/365th of the annual NPP sum
            AVAIL_NPP[i] = AVAIL_NPP[i] / 365
