# cython: language_level=3

'''
SMAP Level 4 Carbon (L4C) heterotrophic respiration calculation, based on
Version 6 state and parameters, at 1-km spatial resolution. The `main()`
routine is optimized for model execution but it may take several seconds to
load the state data.

After the initial state data are loaded it takes about 15-20 seconds per
data day.

Required data:

- Surface soil wetness ("SMSF"), in percentage units [0,100]
- Soil temperature, in degrees K

Developer notes:

- Large datasets (1-km resolution) that are read-in from disk by one function,
    `load_state()`, and read-in from memory by another, `main()`, MUST be
    assigned to global variables because they must use heap allocation.

Possible improvements:

- [ ] Add support for NEE output, by reading in L4C GPP data
- [ ] 1-km global grid files will always be ~500 MB in size, without
    compression; try writing the array to an HDF5 file instead.
'''

import cython
import datetime
import yaml
import numpy as np
from libc.stdio cimport FILE, fopen, fread, fclose, fwrite
from libc.math cimport fmax
from cython.parallel import prange
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from l4cython.respiration cimport is_valid, arrhenius, linear_constraint
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
    # Additional Tsoil parameter (fixed for all PFTs)
    float TSOIL1 = 66.02 # deg K
    float TSOIL2 = 227.13 # deg K
    unsigned char* PFT
    float* SOC0
    float* SOC1
    float* SOC2
    float* LITTERFALL
PFT = <unsigned char*> PyMem_Malloc(sizeof(unsigned char) * SPARSE_M01_N)
SOC0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
SOC1 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
SOC2 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
LITTERFALL  = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)

# Python arrays that want heap allocations must be global; this one is reused
#   for any array that needs to be written to disk (using NumPy)
OUT_M01 = np.full((SPARSE_M01_N,), np.nan, dtype = np.float32)


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
        Py_ssize_t j
        Py_ssize_t k
        Py_ssize_t pft
        float reco # Ecosystem respiration
        char* ofname # Output filename
        float* gpp
        float* smsf
        float* tsoil
        float* rh0
        float* rh1
        float* rh2
        float* rh_total
        float* nee
        float* w_mult
        float* t_mult
        float* soc_total
        float k_mult
    # Note that some datasets are only available at 9-km (M09) resolution
    gpp = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    smsf = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    tsoil = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    rh0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    rh1 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    rh2 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    rh_total = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    nee = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    w_mult = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    t_mult = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    soc_total = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)

    # Read in configuration file, then load state data
    if config is None:
        config = '../data/L4Cython_RECO_M01_config.yaml'
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

    load_state(config) # Load global state variables
    date_start = datetime.datetime.strptime(config['origin_date'], '%Y-%m-%d')
    num_steps = int(config['daily_steps'])
    for step in tqdm(range(num_steps)):
        date = date_start + datetime.timedelta(days = step)
        date = date.strftime('%Y%m%d')
        # Read in soil moisture ("smsf") and soil temperature ("tsoil") data
        fid = open_fid(
            (config['data']['drivers']['smsf'] % date).encode('UTF-8'), READ)
        fread(smsf, sizeof(float), <size_t>sizeof(float)*SPARSE_M09_N, fid)
        fclose(fid)
        fid = open_fid(
            (config['data']['drivers']['tsoil'] % date).encode('UTF-8'), READ)
        fread(tsoil, sizeof(float), <size_t>sizeof(float)*SPARSE_M09_N, fid)
        fclose(fid)
        # Read in the GPP data
        fid = open_fid(
            (config['data']['drivers']['GPP'] % date).encode('UTF-8'), READ)
        fread(gpp, sizeof(float), <size_t>sizeof(float)*SPARSE_M09_N, fid)
        fclose(fid)

        # Iterate over each 9-km pixel
        for i in prange(SPARSE_M09_N, nogil = True):
            # Iterate over each nested 1-km pixel
            for j in range(M01_NESTED_IN_M09):
                # Hence, (i) indexes the 9-km pixel and k the 1-km pixel
                k = (M01_NESTED_IN_M09 * i) + j
                w_mult[k] = FILL_VALUE
                t_mult[k] = FILL_VALUE
                rh_total[k] = FILL_VALUE
                soc_total[k] = FILL_VALUE
                nee[k] = FILL_VALUE
                pft = PFT[k]
                if is_valid(pft, tsoil[i], LITTERFALL[k]) == 0:
                    continue # Skip invalid PFTs
                w_mult[k] = linear_constraint(
                    smsf[i], PARAMS.smsf0[pft], PARAMS.smsf1[pft], 0)
                t_mult[k] = arrhenius(tsoil[i], PARAMS.tsoil[pft], TSOIL1, TSOIL2)
                k_mult = w_mult[k] * t_mult[k]
                rh0[k] = k_mult * SOC0[k] * PARAMS.decay_rate[0][pft]
                rh1[k] = k_mult * SOC1[k] * PARAMS.decay_rate[1][pft]
                rh2[k] = k_mult * SOC2[k] * PARAMS.decay_rate[2][pft]
                # Calculate change in SOC pools; LITTERFALL[k] is daily litterfall
                SOC0[k] += (LITTERFALL[k] * PARAMS.f_metabolic[pft]) - rh0[k]
                SOC1[k] += (LITTERFALL[k] * (1 - PARAMS.f_metabolic[pft])) - rh1[k]
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
                # Compute NEE
                reco = rh_total[k] + (gpp[i] * (1 - PARAMS.cue[pft]))
                nee[k] = reco - gpp[i]

        # If averaging from 1-km to 9-km resolution is requested...
        out_dir = config['model']['output_dir']
        if config['model']['output_format'] in ('M09', 'M09land'):
            fmt = config['model']['output_format']
            inflated = 1 if fmt == 'M09' else 0
            if 'RH' in config['model']['output_fields']:
                output_filename = ('%s/L4Cython_RH_%s_%s.flt32' % (out_dir, date, fmt))\
                    .encode('UTF-8')
                write_resampled(output_filename, rh_total, inflated)
            if 'NEE' in config['model']['output_fields']:
                output_filename = ('%s/L4Cython_NEE_%s_%s.flt32' % (out_dir, date, fmt))\
                    .encode('UTF-8')
                write_resampled(output_filename, nee, inflated)
            if 'Tmult' in config['model']['output_fields']:
                output_filename = ('%s/L4Cython_Tmult_%s_%s.flt32' % (out_dir, date, fmt))\
                    .encode('UTF-8')
                write_resampled(output_filename, t_mult, inflated)
            if 'Wmult' in config['model']['output_fields']:
                output_filename = ('%s/L4Cython_Wmult_%s_%s.flt32' % (out_dir, date, fmt))\
                    .encode('UTF-8')
                write_resampled(output_filename, w_mult, inflated)
        else:
            if 'RH' in config['model']['output_fields']:
                OUT_M01 = to_numpy(rh_total, SPARSE_M01_N)
                OUT_M01.tofile(
                    '%s/L4Cython_RH_%s_M01land.flt32' % (out_dir, date))
            if 'NEE' in config['model']['output_fields']:
                OUT_M01 = to_numpy(nee, SPARSE_M01_N)
                OUT_M01.tofile(
                    '%s/L4Cython_NEE_%s_M01land.flt32' % (out_dir, date))
            if 'Tmult' in config['model']['output_fields']:
                OUT_M01 = to_numpy(t_mult, SPARSE_M01_N)
                OUT_M01.tofile(
                    '%s/L4Cython_Tmult_%s_M01land.flt32' % (out_dir, date))
            if 'Wmult' in config['model']['output_fields']:
                OUT_M01 = to_numpy(w_mult, SPARSE_M01_N)
                OUT_M01.tofile(
                    '%s/L4Cython_Wmult_%s_M01land.flt32' % (out_dir, date))
    PyMem_Free(PFT)
    PyMem_Free(SOC0)
    PyMem_Free(SOC1)
    PyMem_Free(SOC2)
    PyMem_Free(LITTERFALL)
    PyMem_Free(gpp)
    PyMem_Free(smsf)
    PyMem_Free(tsoil)
    PyMem_Free(rh0)
    PyMem_Free(rh1)
    PyMem_Free(rh2)
    PyMem_Free(rh_total)
    PyMem_Free(nee)
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
        LITTERFALL[i] = LITTERFALL[i] / 365


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
    data_resampled.tofile('/home/arthur/temp.flt32')
    # Write a flat (1D) file or inflate the file and then write
    if inflated == 0:
        data_resampled.tofile(output_filename.decode('UTF-8'))
    else:
        write_inflated(output_filename, data_resampled, grid = 'M09')
