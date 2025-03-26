# cython: language_level=3
# distutils: sources = ["utils/src/spland.c", "utils/src/uuta.c"]

'''
SMAP Level 4 Carbon (L4C) heterotrophic respiration (RH) and NEE calculation
at 1-km spatial resolution. The `main()` routine is optimized for model
execution but it may take several seconds to load the state data.

After the initial state data are loaded it takes about 15-30 seconds per
data day when writing one or two fluxes out. Time increases considerably
with more daily output variables.

Required daily driver data:

- SM_SURFACE_WETNESS, in percentage units [0,100]
- SOIL_TEMP_LAYER1, in degrees K
- Gross primary productivity (GPP), in [g C m-2 day-1]

Developer notes:

- Large datasets (1-km resolution) that are read-in from disk by one function,
    `load_state()`, and read-in from memory by another, `main()`, MUST be
    assigned to global variables because they must use heap allocation.

Possible improvements:

- [ ] 1-km global grid files will always be ~500 MB in size, without
    compression; try writing the array to an HDF5 file instead.
'''

import cython
import datetime
import yaml
import numpy as np
import h5py
from libc.stdlib cimport calloc, free
from libc.math cimport fmax
from cython.parallel import prange
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from tempfile import NamedTemporaryFile
from l4cython.core cimport BPLUT, FILL_VALUE, M01_NESTED_IN_M09, SPARSE_M09_N, SPARSE_M01_N, NCOL1KM, NROW1KM, NCOL9KM, NROW9KM, N_PFT, DFNT_FLOAT32
from l4cython.core import load_parameters_table
from l4cython.science cimport arrhenius, linear_constraint
from l4cython.resample cimport write_resampled
from l4cython.utils.hdf5 cimport H5T_STD_U8LE, hid_t, read_hdf5, close_hdf5
from l4cython.utils.io cimport READ, open_fid, read_flat, to_numpy
from l4cython.utils.mkgrid import write_numpy_inflated, write_numpy_deflated
from l4cython.utils.mkgrid cimport deflate, size_in_bytes
from tqdm import tqdm

# EASE-Grid 2.0 params are repeated here to facilitate multiprocessing (they
#   can't be Python numbers)
cdef:
    BPLUT PARAMS
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
LITTERFALL = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)


@cython.boundscheck(False)
@cython.wraparound(False)
def main(config = None, verbose = True):
    '''
    Forward run of the SMAP Level 4 Carbon (L4C) soil decomposition and
    heterotrophic respiration algorithm. Starts on "origin_date" and
    continues for the specified number of time steps.

    Parameters
    ----------
    config : str or dict
    verbose : bool
    '''
    cdef:
        Py_ssize_t i, j, k, pft
        Py_ssize_t doy # Day of year, on [1,365]
        hid_t fid # For open HDF5 files
        int n_litter_days
        float litter # Amount of litterfall entering SOC pools
        float reco # Ecosystem respiration
        float k_mult
    # Note that some datasets are only available at 9-km (M09) resolution
    gpp = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    smsf = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    tsoil = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    litter_rate = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
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
        config = '../data/L4Cython_RECO_config.yaml'
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
        for i in range(0, SPARSE_M01_N):
            litter_rate[i] = 1

    load_state(config) # Load global state variables
    # Begin forward time stepping
    date_start = datetime.datetime.strptime(config['origin_date'], '%Y-%m-%d')
    num_steps = int(config['daily_steps'])
    for step in tqdm(range(num_steps)):
        date = date_start + datetime.timedelta(days = step)
        date_str = date.strftime('%Y%m%d')
        doy = int(date.strftime('%j'))
        # Read in soil moisture ("smsf") and soil temperature ("tsoil") data
        # We re-use a single NamedTemporaryFile(), overwriting its contents
        #   without creating a new instance; there should be little risk in
        #  this because each array is the same size
        tmp = NamedTemporaryFile()
        tmp_fname_bs = tmp.name.encode('UTF-8')
        with h5py.File(
                config['data']['drivers']['file'] % date_str, 'r') as hdf:
            # SMSF
            write_numpy_deflated(tmp_fname_bs, hdf['SM_SURFACE_WETNESS'][:])
            read_flat(tmp_fname_bs, SPARSE_M09_N, smsf)
            # Tsoil
            write_numpy_deflated(tmp_fname_bs, hdf['SOIL_TEMP_LAYER1'][:])
            read_flat(tmp_fname_bs, SPARSE_M09_N, tsoil)

        # Read in the GPP data
        fname_bs = (config['data']['drivers']['GPP'] % date_str).encode('UTF-8')
        read_flat(fname_bs, SPARSE_M09_N, gpp)

        # Option to schedule the rate at which litterfall enters SOC pools
        if config['model']['litterfall']['scheduled']:
            # Get the file covering the 8-day period in which this DOY falls
            filename = config['data']['litterfall_schedule']\
                % str(periods[doy-1]).zfill(2)
            read_flat(filename.encode('UTF-8'), SPARSE_M01_N, litter_rate)

        # Iterate over each 9-km pixel
        for i in prange(SPARSE_M09_N, nogil = True):
            # Iterate over each nested 1-km pixel
            for j in range(M01_NESTED_IN_M09):
                # Hence, (i) indexes the 9-km pixel and k the 1-km pixel
                k = (M01_NESTED_IN_M09 * i) + j
                # Make sure to fill output grids with the FILL_VALUE,
                #   otherwise they may contain zero (0) at invalid data
                w_mult[k] = FILL_VALUE
                t_mult[k] = FILL_VALUE
                rh_total[k] = FILL_VALUE
                soc_total[k] = FILL_VALUE
                nee[k] = FILL_VALUE
                pft = PFT[k]
                if is_valid(pft, tsoil[i], LITTERFALL[k]) == 0:
                    continue # Skip invalid PFTs
                # Compute daily fraction of litterfall entering SOC pools
                litter = LITTERFALL[k] * (fmax(0, litter_rate[k]) / n_litter_days)
                # Compute daily RH based on moisture, temperature constraints;
                #   convert SMSF to % wetness
                w_mult[k] = linear_constraint(
                    smsf[i] * 100.0, PARAMS.smsf0[pft], PARAMS.smsf1[pft], 0)
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
                # Compute NEE
                reco = rh_total[k] + (gpp[i] * (1 - PARAMS.cue[pft]))
                nee[k] = reco - gpp[i]

        # If averaging from 1-km to 9-km resolution is requested...
        fmt = config['model']['output_format']
        suffix = '%s_%s' % (date_str, fmt) # e.g., "*_YYYYMMDD_M09land_*"
        suffix = suffix.encode('UTF-8')
        output_fields = list(map(
            lambda x: x.upper(), config['model']['output_fields']))

        if fmt in ('M09', 'M09land'):
            fid = 0
            inflated = 1 if fmt == 'M09' else 0
            if 'RH' in config['model']['output_fields']:
                fid = write_resampled(config, rh_total, suffix, 'RH', inflated, fid)
            if 'NEE' in config['model']['output_fields']:
                fid = write_resampled(config, nee, suffix, 'NEE', inflated, fid)
            if 'Tmult' in config['model']['output_fields']:
                fid = write_resampled(config, t_mult, suffix, 'Tmult', inflated, fid)
            if 'Wmult' in config['model']['output_fields']:
                fid = write_resampled(config, w_mult, suffix, 'Wmult', inflated, fid)
            if 'SOC' in config['model']['output_fields']:
                fid = write_resampled(config, soc_total, suffix, 'SOC', inflated, fid)
            if config['model']['output_type'].upper() == 'HDF5':
                close_hdf5(fid)
        else:
            output_dir = config['model']['output_dir']
            out_fname_tpl = '%s/L4Cython_%%s_%s_%s.flt32' % (output_dir, date, fmt)
            if 'RH' in config['model']['output_fields']:
                to_numpy(rh_total, SPARSE_M01_N).tofile(out_fname_tpl % 'RH')
            if 'NEE' in config['model']['output_fields']:
                to_numpy(nee, SPARSE_M01_N).tofile(out_fname_tpl % 'NEE')
            if 'Tmult' in config['model']['output_fields']:
                to_numpy(t_mult, SPARSE_M01_N).tofile(out_fname_tpl % 'Tmult')
            if 'Wmult' in config['model']['output_fields']:
                to_numpy(w_mult, SPARSE_M01_N).tofile(out_fname_tpl % 'Wmult')
            if 'SOC' in config['model']['output_fields']:
                to_numpy(soc_total, SPARSE_M01_N).tofile(out_fname_tpl % 'SOC')

    PyMem_Free(PFT)
    PyMem_Free(LITTERFALL)
    PyMem_Free(SOC0)
    PyMem_Free(SOC1)
    PyMem_Free(SOC2)
    PyMem_Free(gpp)
    PyMem_Free(smsf)
    PyMem_Free(tsoil)
    PyMem_Free(litter_rate)
    PyMem_Free(rh0)
    PyMem_Free(rh1)
    PyMem_Free(rh2)
    PyMem_Free(rh_total)
    PyMem_Free(nee)
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
    ancillary_file_bs = config['data']['ancillary']['file'].encode('UTF-8')
    pft_map_field = config['data']['ancillary']['PFT_map'].encode('UTF-8')
    read_hdf5(ancillary_file_bs, pft_map_field, H5T_STD_U8LE, PFT)

    # SOC and NPP are found in the restart file
    restart_file_bs = config['data']['restart']['file'].encode('UTF-8')
    # Read in SOC datasets
    soc_fields = config['data']['restart']['SOC']
    # We re-use a single NamedTemporaryFile(), overwriting its contents
    #   without creating a new instance; there should be little risk in
    #  this because each array is the same size
    tmp = NamedTemporaryFile()
    tmp_fname_bs = tmp.name.encode('UTF-8')
    with h5py.File(restart_file_bs, 'r') as hdf:
        write_numpy_deflated(
            tmp_fname_bs, hdf[soc_fields[0]][:], grid = 'M01')
        read_flat(tmp_fname_bs, SPARSE_M01_N, SOC0)
        write_numpy_deflated(
            tmp_fname_bs, hdf[soc_fields[1]][:], grid = 'M01')
        read_flat(tmp_fname_bs, SPARSE_M01_N, SOC1)
        write_numpy_deflated(
            tmp_fname_bs, hdf[soc_fields[2]][:], grid = 'M01')
        read_flat(tmp_fname_bs, SPARSE_M01_N, SOC2)
        # Now, for litterfall (annual sum of NPP)
        npp_field = config['data']['restart']['NPP']
        write_numpy_deflated(
            tmp_fname_bs, hdf[npp_field][:], grid = 'M01')
        read_flat(tmp_fname_bs, SPARSE_M01_N, LITTERFALL)

    # NOTE: Calculating litterfall as average daily NPP (constant fraction of
    #   the annual NPP sum)
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
