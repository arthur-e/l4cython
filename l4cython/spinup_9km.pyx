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

Note that if the "model/ending_day_of_year" configuration parameter is set to
anything less than 365 (December 31), the model will run an additional
climatological cycle, up to and ending on the day of year (DOY) specified, in
order to align with the desired the seasonal cycle.

Recent benchmarks on Gullveig (Intel Xeon 3.7 GHz): 10-50 secs (depending on
NFS latency) for analytical spin-up, 25-30 min for the first climatological
year in numerical spin-up. It should get faster for each successive
climatological year.

Required data:

- Surface soil wetness ("SMSF"), in percentage units [0,100]
- Soil temperature, in degrees K
'''

import cython
import datetime
import yaml
import numpy as np
from libc.math cimport isnan, fabs, fmax
from libc.stdio cimport FILE, fread, fclose
from cython.parallel import prange
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from l4cython.constraints cimport arrhenius, linear_constraint
from l4cython.core cimport BPLUT, SPARSE_M09_N, N_PFT
from l4cython.core import load_parameters_table
from l4cython.utils.io cimport READ, open_fid, to_numpy_double
from tqdm import tqdm

cdef:
    FILE* fid
    BPLUT PARAMS
    int SPARSE_N = SPARSE_M09_N # Number of grid cells in sparse ("land") arrays
    # Additional Tsoil parameter (fixed for all PFTs)
    float TSOIL1 = 66.02 # deg K
    float TSOIL2 = 227.13 # deg K
    unsigned char* PFT
    float* AVAIL_NPP
PFT = <unsigned char*> PyMem_Malloc(sizeof(unsigned char) * SPARSE_N)
AVAIL_NPP = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)

# Python arrays that want heap allocations must be global; this one is reused
#   for any array that needs to be written to disk (using NumPy)
OUT_M09_DOUBLE = np.full((SPARSE_N,), np.nan, dtype = np.float64)


@cython.boundscheck(False)
@cython.wraparound(False)
def main(config = None, verbose = True):
    '''
    Soil organic carbon (SOC) spin-up; will write out a file for each soil
    pool after the analytical spin-up ("Cana") and after the numerical
    spin-up ("Cnum"), in addition to a final tolerance file.

    Parameters
    ----------
    config : str or dict
    verbose : bool
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
    if config is None:
        config = '../data/L4Cython_spin-up_M09_config.yaml'
    if isinstance(config, str):
        with open(config, 'r') as file:
            config = yaml.safe_load(file)

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

    # Load global state variables
    load_state(config)

    # Step 1: Analytical spin-up; note that soc0, soc1, soc2 are both
    #   inputs and outputs of the spin-up functions (K&R-style)
    analytical_spinup(config, soc0, soc1, soc2, 1 if verbose else 0)
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
    numerical_spinup(config, soc0, soc1, soc2, 1 if verbose else 0)
    output_dir = config['model']['output_dir']
    end_doy = str(config['model']['ending_day_of_year']).zfill(3)
    OUT_M09_DOUBLE = to_numpy_double(soc0, SPARSE_N)
    if config['model']['accelerated']:
        OUT_M09_DOUBLE *= config['model']['ad_rate'][0]
    OUT_M09_DOUBLE.tofile(
        f'{output_dir}/L4Cython_Cnum0_M09land_DOY{end_doy}.flt64')
    OUT_M09_DOUBLE = to_numpy_double(soc1, SPARSE_N)
    if config['model']['accelerated']:
        OUT_M09_DOUBLE *= config['model']['ad_rate'][1]
    OUT_M09_DOUBLE.tofile(
        f'{output_dir}/L4Cython_Cnum1_M09land_DOY{end_doy}.flt64')
    OUT_M09_DOUBLE = to_numpy_double(soc2, SPARSE_N)
    if config['model']['accelerated']:
        OUT_M09_DOUBLE *= config['model']['ad_rate'][2]
    OUT_M09_DOUBLE.tofile(
        f'{output_dir}/L4Cython_Cnum2_M09land_DOY{end_doy}.flt64')
    PyMem_Free(soc0)
    PyMem_Free(soc1)
    PyMem_Free(soc2)


cdef analytical_spinup(
        config, double* soc0, double* soc1, double* soc2, int verbose):
    '''
    Analytical SOC spin-up: the initial soil C states are found by solving
    the differential equations that describe inputs, transfers, and decay
    of soil C.

    Parameters
    ----------
    config : dict
    soc0 : double*
    soc1 : double*
    soc2 : double*
    verbose : int
    '''
    cdef:
        Py_ssize_t i
        Py_ssize_t doy
        int accelerated, pft, n_litter_days
        float ad_rate[3]
        float* smsf
        float* tsoil
        float* litter # Usually, this is annual total NPP
        float* litter_rate # Fraction of litterfall allocated
        float* k0 # Decay rates of each pool
        float* k1
        float* k2
        float* k_mult
        float* w_mult
        float* t_mult
        float denom0, denom1
    smsf = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    tsoil = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    litter = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    litter_rate = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    k0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    k1 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    k2 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    k_mult = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    w_mult = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    t_mult = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)

    # Accommodate accelerated decomposition
    accelerated = 0
    if config['model']['accelerated']:
        accelerated = 1
        for i in range(3):
            ad_rate[i] = config['model']['ad_rate'][i]

    # Option to schedule the rate at which litterfall enters SOC pools; if no
    #   schedule is used, an equal daily fraction of available NPP allocated
    if config['model']['litterfall']['scheduled']:
        n_litter_days = config['model']['litterfall']['interval_days']
        n_litter_periods = np.ceil(365 / n_litter_days)
        periods = np.array([
            [i] * n_litter_days for i in range(1, n_litter_periods + 1)
        ]).ravel()
    else:
        n_litter_days = 1
        for i in prange(0, SPARSE_N, nogil = True):
            litter_rate[i] = 1/365.0 # Allocate equal daily fraction

    if verbose == 1:
        print('Beginning analytical spin-up...')
    # Analytical spin-up calculation has a closed form, so this outer loop,
    #   for each day of the climatological year, pre-computes annual sums
    for doy in tqdm(range(1, 366), disable = verbose == 0):

        jday = str(doy).zfill(3)
        # NOTE: It will ALWAYS be faster to read these all-at-once rather than
        #   to initialize them with heap allocations, then extract each
        #   element one-at-a-time (as in load_state())
        fid = open_fid(
            (config['data']['climatology']['tsoil'] % jday).encode('UTF-8'), READ)
        fread(tsoil, sizeof(float), <size_t>sizeof(float)*SPARSE_N, fid)
        fclose(fid)
        # NOTE: For compatibility with TCF, the SMSF data are already in
        #   percentage units, i.e., on [0,100]
        fid = open_fid(
            (config['data']['climatology']['smsf'] % jday).encode('UTF-8'), READ)
        fread(smsf, sizeof(float), <size_t>sizeof(float)*SPARSE_N, fid)
        fclose(fid)

        # Option to schedule the rate at which litterfall enters SOC pools
        if config['model']['litterfall']['scheduled']:
            # Get the file covering the 8-day period in which this DOY falls
            filename = config['data']['litterfall_schedule']\
                % str(periods[doy-1]).zfill(2)
            fid = open_fid(filename.encode('UTF-8'), READ)
            fread(litter_rate, sizeof(float), <size_t>sizeof(float)*SPARSE_N, fid)
            fclose(fid)

        # Pre-compute the k_mult sum for each pixel
        for i in prange(SPARSE_N, nogil = True):
            pft = int(PFT[i])
            if pft == 0 or pft > 8:
                continue
            # Just once for each pixel
            if doy == 1:
                litter[i] = 0
                # Initialize k_mult sum at zero; set base decay rates and
                #   allocation factors
                k_mult[i] = 0
                k0[i] = PARAMS.decay_rate[0][pft]
                k1[i] = PARAMS.decay_rate[1][pft]
                k2[i] = PARAMS.decay_rate[2][pft]
                if accelerated:
                    k0[i] = k0[i] * ad_rate[0]
                    k1[i] = k1[i] * ad_rate[1]
                    k2[i] = k2[i] * ad_rate[2]
            # Total annual NPP; if litterfall is not scheduled, this is just
            #   a sum of 365 equal increments
            litter[i] += (
                AVAIL_NPP[i] * fmax(0, litter_rate[i] / n_litter_days))
            # Compute and increment annual k_mult sum
            w_mult[i] = linear_constraint(
                smsf[i], PARAMS.smsf0[pft], PARAMS.smsf1[pft], 0)
            t_mult[i] = arrhenius(tsoil[i], PARAMS.tsoil[pft], TSOIL1, TSOIL2)
            k_mult[i] += (w_mult[i] * t_mult[i])

    # Outside of the daily loop, but for each pixel...
    for i in prange(SPARSE_N, nogil = True):
        pft = int(PFT[i])
        if pft == 0 or pft > 8:
            continue
        # Calculate analytical steady-state of SOC
        denom0 = (k_mult[i] * k0[i])
        if denom0 > 0:
            soc0[i] = <double>(litter[i] * PARAMS.f_metabolic[pft]) / denom0
        else:
            soc0[i] = 0
        denom1 = (k_mult[i] * k1[i])
        if denom1 > 0:
            soc1[i] = <double>(litter[i] * (1 - PARAMS.f_metabolic[pft])) / denom1
        else:
            soc1[i] = 0
        # Guard against division by zero
        if k2[i] > 0:
            soc2[i] = <double>(PARAMS.f_structural[pft] * k1[i] * soc1[i]) / k2[i]
        else:
            soc2[i] = 0
    PyMem_Free(smsf)
    PyMem_Free(tsoil)
    PyMem_Free(litter)
    PyMem_Free(litter_rate)
    PyMem_Free(k0)
    PyMem_Free(k1)
    PyMem_Free(k2)
    PyMem_Free(k_mult)
    PyMem_Free(w_mult)
    PyMem_Free(t_mult)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef numerical_spinup(
        config, double* soc0, double* soc1, double* soc2, int verbose):
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
        int iter, pft, n_litter_days
        int tol_count # Number of pixels with tolerance, for calculating...
        float tol_mean # ...Overall mean tolerance
        float tol_sum # Sum of all tolerances
        float litterfall
        float w_mult, t_mult
        float ad_rate[3]
        float* smsf
        float* tsoil
        float* litter_rate # Fraction of litterfall allocated
        float* rh_total # Annual heterotrophic respiration (RH) sum
        float* nee_sum # Annual NEE sum
        float* nee_last_year # Last year's annual NEE sum
        double* delta # 3-element, recycling vector: diff. in each pool
        double* tolerance # Tolerance at each pixel
    smsf = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    tsoil = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    litter_rate = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    rh_total = <float*> PyMem_Malloc(sizeof(float) * 1)
    nee_sum = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    nee_last_year = <float*> PyMem_Malloc(sizeof(float) * SPARSE_N)
    delta = <double*> PyMem_Malloc(sizeof(double) * 3)
    tolerance = <double*> PyMem_Malloc(sizeof(double) * SPARSE_N)

    # Accommodate accelerated decomposition
    for i in range(3):
        if config['model']['accelerated']:
            ad_rate[i] = config['model']['ad_rate'][i]
        else:
            ad_rate[i] = 1

    # Option to schedule the rate at which litterfall enters SOC pools; if no
    #   schedule is used, an equal daily fraction of available NPP allocated
    if config['model']['litterfall']['scheduled']:
        n_litter_days = config['model']['litterfall']['interval_days']
        n_litter_periods = np.ceil(365 / n_litter_days)
        periods = np.array([
            [i] * n_litter_days for i in range(1, n_litter_periods + 1)
        ]).ravel()
    else:
        n_litter_days = 1
        for i in prange(0, SPARSE_N, nogil = True):
            litter_rate[i] = 1/365.0 # Allocate equal daily fraction

    # Pre-allocate tolerance array
    for i in range(SPARSE_N):
        tolerance[i] = 0

    if verbose == 1:
        print('Beginning numerical spin-up...')
    end_doy = 366 # Normally, we run a full climatological year
    # This is where we want to end up in the seasonal cycle:
    projected_end_doy = config['model']['ending_day_of_year']
    min_pixels = config['min_pixels'] # Minimum number of pixels remaining
    max_iter = config['max_iter']
    iter = 1
    equilibrating = True
    while equilibrating:
        # If there's a stopping criterion that's met, make sure we wind up
        #   at the right part of the seasonal cycle; criteria are:
        #   1) Performed the maximum number of climatological cycles (iter)
        #   2) Number of pixels that meet tolerance is less than the minimum
        if iter > 1 and (iter >= max_iter or tol_count <= min_pixels):
            # We do this check here so that the while loop runs one more time
            #   after successful equilibration, to align seasonal cycle
            equilibrating = False
            # But we can quit now if we intended to end on December 31
            if projected_end_doy == 365:
                break
            else:
                end_doy = projected_end_doy + 1

        tol_sum = 0.0 # Reset sum (for calculating average) of tolerance
        tol_count = 0 # Reset count of pixels that have yet to equilibrate
        # For each day of the climatological year
        for doy in tqdm(range(1, end_doy), disable = verbose == 0):
            jday = str(doy).zfill(3)
            # NOTE: For compatibility with TCF, the SMSF data are already in
            #   percentage units, i.e., on [0,100]
            fid = open_fid(
                (config['data']['climatology']['smsf'] % jday).encode('UTF-8'), READ)
            fread(smsf, sizeof(float), <size_t>sizeof(float)*SPARSE_N, fid)
            fclose(fid)
            fid = open_fid(
                (config['data']['climatology']['tsoil'] % jday).encode('UTF-8'), READ)
            fread(tsoil, sizeof(float), <size_t>sizeof(float)*SPARSE_N, fid)
            fclose(fid)

            # Option to schedule the rate at which litterfall enters SOC pools
            if config['model']['litterfall']['scheduled']:
                # Get the file covering the 8-day period in which this DOY falls
                filename = config['data']['litterfall_schedule']\
                    % str(periods[doy-1]).zfill(2)
                fid = open_fid(filename.encode('UTF-8'), READ)
                fread(litter_rate, sizeof(float), <size_t>sizeof(float)*SPARSE_N, fid)
                fclose(fid)

            # Loop over pixels
            for i in prange(SPARSE_N, nogil = True):
                pft = int(PFT[i])
                if pft == 0 or pft > 8:
                    continue
                # Skip pixels that have already equilibrated
                if iter > 1 and not isnan(tolerance[i]):
                    # Jones et al. (2017) write that goal is NEE
                    #   tolerance <= 1 g C m-2 year-1
                    if fabs(tolerance[i]) < 1:
                        continue
                # Reset the annual delta-SOC arrays
                delta[0] = 0
                delta[1] = 0
                delta[2] = 0
                # Calculate the litterfall input this time step: a certain
                #   fraction of the annual NPP sum
                litterfall = (
                    AVAIL_NPP[i] * fmax(0, litter_rate[i] / n_litter_days))
                w_mult = linear_constraint(
                    smsf[i], PARAMS.smsf0[pft], PARAMS.smsf1[pft], 0)
                t_mult = arrhenius(tsoil[i], PARAMS.tsoil[pft], TSOIL1, TSOIL2)
                # Compute one daily soil decomposition step for this pixel;
                #   note that litter[i] is the daily litterfall
                numerical_step(
                    delta, rh_total, litterfall, w_mult * t_mult,
                    PARAMS.f_metabolic[pft], PARAMS.f_structural[pft],
                    PARAMS.decay_rate[0][pft], PARAMS.decay_rate[1][pft],
                    PARAMS.decay_rate[2][pft], soc0[i], soc1[i], soc2[i],
                    ad_rate)
                # Compute change in SOC storage as SOC + dSOC, it's unclear
                #   why, but we absolutely MUST test for NaNs here, not upstream
                if not isnan(delta[0] + delta[1] + delta[2]):
                    soc0[i] += delta[0]
                    soc1[i] += delta[1]
                    soc2[i] += delta[2]
                # At end of year, calculate change in Delta-NEE relative to
                #   previous year
                if doy == 365:
                    # i.e., tolerance is the change in the Annual NEE sum:
                    #   NEE = (RA + RH) - GPP --> NEE = RH - NPP
                    #   DeltaNEE = NEE(t) - NEE(t-1)
                    nee_sum[i] = rh_total[0] - AVAIL_NPP[i]
                    if iter > 0:
                        tolerance[i] = nee_last_year[i] - nee_sum[i]
                    else:
                        tolerance[i] = nee_sum[i]
                    nee_last_year[i] = nee_sum[i]
                    if not isnan(tolerance[i]):
                        tol_count += 1
                        tol_sum += tolerance[i]

        if verbose == 1:
            print('[%d/%d] Total tolerance is: %.2f' % (iter, max_iter, tol_sum))
            print('--- Pixels counted: %d' % tol_count)
        if tol_count > 0:
            tol_mean = (tol_sum / tol_count)
            if verbose == 1:
                print('--- Mean tolerance is: %.2f' % tol_mean)
        # Increment; also a counter for the number of climatological years
        iter = iter + 1

    OUT_M09_DOUBLE = to_numpy_double(tolerance, SPARSE_N)
    OUT_M09_DOUBLE.tofile(
        '%s/L4Cython_numspin_tol_M09land.flt64' % config['model']['output_dir'])
    PyMem_Free(smsf)
    PyMem_Free(tsoil)
    PyMem_Free(litter_rate)
    PyMem_Free(rh_total)
    PyMem_Free(nee_sum)
    PyMem_Free(nee_last_year)
    PyMem_Free(delta)
    PyMem_Free(tolerance)


cdef inline int numerical_step(
        double* delta, float* rh_total, float litter, float k_mult,
        float f_met, float f_str, float k0, float k1, float k2,
        double c0, double c1, double c2, float[3] ad_rate) nogil:
    'A single daily soil decomposition step (for a single pixel)'
    cdef:
        float rh0
        float rh1
        float rh2
    # If not using accelerated decomposition, ad_rate[p] would be ==1
    rh0 = k_mult * k0 * c0 * ad_rate[0]
    rh1 = k_mult * k1 * c1 * ad_rate[1]
    rh2 = k_mult * k2 * c2 * ad_rate[2]
    # Calculate change in C pools (g C m-2 units)
    if litter >= 0:
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
    cdef Py_ssize_t i
    n_bytes = sizeof(unsigned char)*SPARSE_N
    fid = open_fid(config['data']['PFT_map'].encode('UTF-8'), READ)
    fread(PFT, sizeof(unsigned char), <size_t>n_bytes, fid)
    fclose(fid)
    fid = open_fid(config['data']['NPP_annual_sum'].encode('UTF-8'), READ)
    fread(AVAIL_NPP, sizeof(float), <size_t>n_bytes, fid)
    fclose(fid)
    for i in prange(SPARSE_N, nogil = True):
        # Set any negative values (really just -9999) to zero
        AVAIL_NPP[i] = fmax(0, AVAIL_NPP[i])
