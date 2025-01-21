# cython: language_level=3

'''
SMAP Level 4 Carbon (L4C) heterotrophic respiration (RH) and NEE calculation
at 9-km spatial resolution. The `main()` routine is optimized for model
execution but it may take several seconds to load the state data.

Recent benchmarks on Gullveig (Intel Xeon 3.7 GHz): About 1.2s per data-day
when producing a flat ("M09land") output.

Required daily driver data:

- Surface soil wetness ("SMSF"), in percentage units [0,100]
- Soil temperature, in degrees K
- Gross primary productivity (GPP), in [g C m-2 day-1]
'''

import cython
import datetime
import yaml
import numpy as np
from libc.math cimport fmax
from libc.stdio cimport FILE, fread, fclose
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from l4cython.constraints cimport arrhenius, linear_constraint
from l4cython.utils cimport BPLUT, open_fid, to_numpy
from l4cython.utils.mkgrid import write_numpy_inflated
from l4cython.utils.fixtures import READ, SPARSE_M09_N, N_PFT, load_parameters_table
from tqdm import tqdm

cdef:
    FILE* fid
    BPLUT PARAMS
    # Additional Tsoil parameter (fixed for all PFTs)
    float TSOIL1 = 66.02 # deg K
    float TSOIL2 = 227.13 # deg K
    unsigned char* PFT # The PFT map
    float* SOC0
    float* SOC1
    float* SOC2
    float* LITTERFALL
PFT = <unsigned char*> PyMem_Malloc(sizeof(unsigned char) * SPARSE_M09_N)
SOC0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
SOC1 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
SOC2 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
LITTERFALL = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)

# Python arrays that want heap allocations must be global; this one is reused
#   for any array that needs to be written to disk (using NumPy)
OUT_M09 = np.full((SPARSE_M09_N,), np.nan, dtype = np.float32)


@cython.boundscheck(False)
@cython.wraparound(False)
def main(config = None, verbose = True):
    '''
    Forward run of the L4C soil decomposition and heterotrophic respiration
    algorithm. Starts on "origin_date" and continues for the specified number
    of time steps.

    Parameters
    ----------
    config : str or dict
    verbose : bool
    '''
    cdef:
        Py_ssize_t i
        Py_ssize_t doy # Day of year, on [1,365]
        int n_litter_days
        float litter # Amount of litterfall entering SOC pools
        float reco # Ecosystem respiration
        float* litter_rate # Fraction of litterfall allocated
        float* rh0
        float* rh1
        float* rh2
        float* rh_total
        float* gpp
        float* nee
        float* w_mult
        float* t_mult
        float* soil_organic_carbon
    litter_rate = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    rh0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    rh1 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    rh2 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    rh_total = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    gpp = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    nee = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    w_mult = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    t_mult = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    soil_organic_carbon = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)

    # Read in configuration file, then load state data
    if config is None:
        config = '../data/L4Cython_RECO_M09_config.yaml'
    if isinstance(config, str) and verbose:
        print(f'Using config file: {config}')
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

    # Option to schedule the rate at which litterfall enters SOC pools; if no
    #   schedule is used, an equal daily fraction of available NPP allocated
    if config['model']['litterfall']['scheduled']:
        n_litter_days = config['model']['litterfall']['interval_days']
        n_litter_periods = np.ceil(365 / n_litter_days)
        periods = np.array([
            [i] * n_litter_days for i in range(1, n_litter_periods + 1)
        ]).ravel()
    else:
        # Allocate equal daily fraction; i.e., final rate is
        #   (litter_rate / n_litter_days) or 1/365
        n_litter_days = 365
        for i in range(0, SPARSE_M09_N):
            litter_rate[i] = 1

    load_state(config) # Load global state variables
    date_start = datetime.datetime.strptime(config['origin_date'], '%Y-%m-%d')
    num_steps = int(config['daily_steps'])
    for step in tqdm(range(num_steps), disable = not verbose):
        date = date_start + datetime.timedelta(days = step)
        date_str = date.strftime('%Y%m%d')
        doy = int(date.strftime('%j'))
        smsf = np.fromfile(
            config['data']['drivers']['smsf'] % date_str, dtype = np.float32)
        tsoil = np.fromfile(
            config['data']['drivers']['tsoil'] % date_str, dtype = np.float32)
        # Read in the GPP data
        fid = open_fid((config['data']['drivers']['GPP'] % date_str)\
            .encode('UTF-8'), READ)
        fread(gpp, sizeof(float), <size_t>sizeof(float)*SPARSE_M09_N, fid)
        fclose(fid)

        # Option to schedule the rate at which litterfall enters SOC pools
        if config['model']['litterfall']['scheduled']:
            # Get the file covering the 8-day period in which this DOY falls
            filename = config['data']['litterfall_schedule']\
                % str(periods[doy-1]).zfill(2)
            fid = open_fid(filename.encode('UTF-8'), READ)
            fread(litter_rate, sizeof(float), <size_t>sizeof(float)*SPARSE_M09_N, fid)
            fclose(fid)

        for i in range(0, SPARSE_M09_N):
            pft = int(PFT[i])
            if pft not in (1, 2, 3, 4, 5, 6, 7, 8):
                continue
            # Compute daily fraction of litterfall entering SOC pools
            litter = LITTERFALL[i] * (fmax(0, litter_rate[i]) / n_litter_days)
            # Compute daily RH based on moisture, temperature constraints
            w_mult[i] = linear_constraint(
                smsf[i], PARAMS.smsf0[pft], PARAMS.smsf1[pft], 0)
            t_mult[i] = arrhenius(tsoil[i], PARAMS.tsoil[pft], TSOIL1, TSOIL2)
            rh0[i] = w_mult[i] * t_mult[i] * SOC0[i] * PARAMS.decay_rate[0][pft]
            rh1[i] = w_mult[i] * t_mult[i] * SOC1[i] * PARAMS.decay_rate[1][pft]
            rh2[i] = w_mult[i] * t_mult[i] * SOC2[i] * PARAMS.decay_rate[2][pft]
            # Calculate change in SOC pools
            SOC0[i] += (litter * PARAMS.f_metabolic[pft]) - rh0[i]
            SOC1[i] += (litter * (1 - PARAMS.f_metabolic[pft])) - rh1[i]
            SOC2[i] += (PARAMS.f_structural[pft] * rh1[i]) - rh2[i]
            # "the adjustment...to account for material transferred into the
            #   slow pool during humification" (Jones et al. 2017 TGARS, p.5)
            rh1[i] = rh1[i] * (1 - PARAMS.f_structural[pft])
            rh_total[i] = rh0[i] + rh1[i] + rh2[i]
            # Adjust pools, if needed, to guard against negative values
            rh_total[i] = fmax(rh_total[i], 0)
            SOC0[i] = fmax(SOC0[i], 0)
            SOC1[i] = fmax(SOC1[i], 0)
            SOC2[i] = fmax(SOC2[i], 0)
            # Compute NEE
            reco = rh_total[i] + (gpp[i] * (1 - PARAMS.cue[pft]))
            nee[i] = reco - gpp[i]

        # Write datasets to disk
        fname = '%s/L4Cython_{what}_%s_M09.flt32' % (
            config['model']['output_dir'], date_str)
        if config['model']['output_format'] == 'M09land':
            if 'RH' in config['model']['output_fields']:
                OUT_M09 = to_numpy(rh_total, SPARSE_M09_N)
                OUT_M09.tofile(
                    fname.format(what = 'RH').replace('M09', 'M09land'))
            if 'NEE' in config['model']['output_fields']:
                OUT_M09 = to_numpy(nee, SPARSE_M09_N)
                OUT_M09.tofile(
                    fname.format(what = 'NEE').replace('M09', 'M09land'))
            if 'SOC' in config['model']['output_fields']:
                for i in range(0, SPARSE_M09_N):
                    soil_organic_carbon[i] = SOC0[i] + SOC1[i] + SOC2[i]
                OUT_M09 = to_numpy(soil_organic_carbon, SPARSE_M09_N)
                OUT_M09.tofile(
                    fname.format(what = 'SOC').replace('M09', 'M09land'))
            if 'Tmult' in config['model']['output_fields']:
                OUT_M09 = to_numpy(t_mult, SPARSE_M09_N)
                OUT_M09.tofile(
                    fname.format(what = 'Tmult').replace('M09', 'M09land'))
            if 'Wmult' in config['model']['output_fields']:
                OUT_M09 = to_numpy(t_mult, SPARSE_M09_N)
                OUT_M09.tofile(
                    fname.format(what = 'Wmult').replace('M09', 'M09land'))
        else:
            if 'RH' in config['model']['output_fields']:
                write_numpy_inflated(
                    fname.format(what = 'RH').encode('UTF-8'), to_numpy(rh_total, SPARSE_M09_N))
            if 'NEE' in config['model']['output_fields']:
                write_numpy_inflated(
                    fname.format(what = 'NEE').encode('UTF-8'), to_numpy(nee, SPARSE_M09_N))
    # Finally, write out the final SOC state, if we're in debug mode
    if config['debug']:
        fname_soc = '%s/L4Cython_SOC_C{i}_M09.flt32' % config['model']['output_dir']
        if config['model']['output_format'] == 'M09land':
            fname_soc = fname_soc.replace('M09', 'M09land')
        OUT_M09 = to_numpy(SOC0, SPARSE_M09_N)
        OUT_M09.tofile(fname_soc.format(i = 0))
        OUT_M09 = to_numpy(SOC1, SPARSE_M09_N)
        OUT_M09.tofile(fname_soc.format(i = 1))
        OUT_M09 = to_numpy(SOC2, SPARSE_M09_N)
        OUT_M09.tofile(fname_soc.format(i = 2))
    PyMem_Free(PFT)
    PyMem_Free(SOC0)
    PyMem_Free(SOC1)
    PyMem_Free(SOC2)
    PyMem_Free(LITTERFALL)
    PyMem_Free(litter_rate)
    PyMem_Free(rh0)
    PyMem_Free(rh1)
    PyMem_Free(rh2)
    PyMem_Free(rh_total)
    PyMem_Free(gpp)
    PyMem_Free(nee)
    PyMem_Free(w_mult)
    PyMem_Free(t_mult)
    PyMem_Free(soil_organic_carbon)


def load_state(config):
    '''
    Populates global state variables with data.

    Parameters
    ----------
    config : dict
        The configuration data dictionary
    '''
    # Allocate space, read in 1-km PFT map
    fid = open_fid(config['data']['PFT_map'].encode('UTF-8'), READ)
    fread(PFT, sizeof(unsigned char), <size_t>sizeof(unsigned char)*SPARSE_M09_N, fid)
    fclose(fid)
    # Read in SOC datasets
    fid = open_fid(config['data']['SOC'][0].encode('UTF-8'), READ)
    fread(SOC0, sizeof(float), <size_t>sizeof(float)*SPARSE_M09_N, fid)
    fclose(fid)
    fid = open_fid(config['data']['SOC'][1].encode('UTF-8'), READ)
    fread(SOC1, sizeof(float), <size_t>sizeof(float)*SPARSE_M09_N, fid)
    fclose(fid)
    fid = open_fid(config['data']['SOC'][2].encode('UTF-8'), READ)
    fread(SOC2, sizeof(float), <size_t>sizeof(float)*SPARSE_M09_N, fid)
    fclose(fid)
    fid = open_fid(config['data']['NPP_annual_sum'].encode('UTF-8'), READ)
    fread(LITTERFALL, sizeof(float), <size_t>sizeof(float)*SPARSE_M09_N, fid)
    fclose(fid)
    for i in range(SPARSE_M09_N):
        # Set any negative values (really just -9999) to zero
        SOC0[i] = fmax(0, SOC0[i])
        SOC1[i] = fmax(0, SOC1[i])
        SOC2[i] = fmax(0, SOC2[i])
        LITTERFALL[i] = fmax(0, LITTERFALL[i])
