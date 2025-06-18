# cython: language_level=3
# distutils: sources = ["l4cython/utils/src/spland.c", "l4cython/utils/src/uuta.c"]

'''
SMAP Level 4 Carbon (L4C) heterotrophic respiration (RH) and NEE calculation
at 1-km spatial resolution. The `main()` routine is optimized for model
execution but it may take several seconds to load the state data.

After the initial state data are loaded it takes about 15-30 seconds per
data day when writing one or two fluxes out. Time increases considerably
with more daily output variables.

Required daily driver data:

- Surface soil wetness ("SMSF"), in percentage units [0,100]
- Soil temperature, in degrees K

Developer notes:

- Currently requires about 12 GB of memory.
- Large datasets (1-km resolution) that are read-in from disk by one function,
    `load_state()`, and read-in from memory by another, `main()`, MUST be
    assigned to global variables because they must use heap allocation.
'''

import os
import cython
import datetime
import yaml
import numpy as np
import l4cython
from bisect import bisect_right
from libc.stdlib cimport calloc, free
from libc.stdio cimport FILE, fopen, fread, fclose
from libc.math cimport fmax
from cython.parallel import prange
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from tempfile import NamedTemporaryFile
from l4cython.core cimport BPLUT, FILL_VALUE, M01_NESTED_IN_M09, SPARSE_M09_N, SPARSE_M01_N, NCOL1KM, NROW1KM, NCOL9KM, NROW9KM, N_PFT, DFNT_UINT8, DFNT_FLOAT32
from l4cython.core import load_parameters_table
from l4cython.science cimport arrhenius, linear_constraint, rescale_smrz, vapor_pressure_deficit, photosynth_active_radiation
from l4cython.resample cimport write_resampled, write_fullres
from l4cython.utils.dec2bin cimport bits_from_uint32
from l4cython.utils.hdf5 cimport H5T_STD_U8LE, H5T_IEEE_F32LE, hid_t, read_hdf5
from l4cython.utils.io cimport READ, open_fid, write_flat, read_flat, read_flat_short, to_numpy
from l4cython.utils.mkgrid import inflate_file, write_numpy_inflated
from l4cython.utils.mkgrid cimport deflate, size_in_bytes
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
    float* SMRZ_MAX
    float* SMRZ_MIN
PFT = <unsigned char*> PyMem_Malloc(sizeof(unsigned char) * SPARSE_M01_N)
LITTERFALL  = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
SOC0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
SOC1 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
SOC2 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
SMRZ_MAX = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
SMRZ_MIN = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)

# 8-day composite periods, as ordinal days of a 365-day year
PERIODS = np.arange(1, 365, 8)
PERIODS_DATES = [
    datetime.datetime.strptime('2005-%03d' % p, '%Y-%j') for p in PERIODS
]
PERIODS_DATES_LEAP = [
    datetime.datetime.strptime('2004-%03d' % p, '%Y-%j') for p in PERIODS
]


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
        int check_fpar_qc # =1 if fPAR QC should be checked
        float fpar # Current fPAR value
        float litter # Amount of litterfall entering SOC pools
        float reco # Ecosystem respiration
        float k_mult

    # 9-km (M09) heap allocations, for RECO
    smsf  = <short int*> PyMem_Malloc(sizeof(short int) * SPARSE_M09_N)
    tsoil = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    # For GPP
    smrz0 = <short int*> PyMem_Malloc(sizeof(short int) * SPARSE_M09_N)
    smrz  = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    swrad = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    t2m   = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    tmin  = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    qv2m  = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    ps    = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    tsurf = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    vpd   = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    par   = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    # 1-km (M01) heap allocations, for GPP
    litter_rate = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    rh0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    rh1 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    rh2 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    rh_total = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    nee = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    w_mult = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    t_mult = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    soc_total = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    # For GPP
    ft    = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    e_mult= <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    gpp = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    npp = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    f_tmin = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    f_vpd  = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    f_smrz = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    fpar0 = <unsigned char*> PyMem_Malloc(sizeof(unsigned char) * SPARSE_M01_N)
    fpar_qc = <unsigned char*> PyMem_Malloc(sizeof(unsigned char) * SPARSE_M01_N)
    fpar_clim = <unsigned char*> PyMem_Malloc(sizeof(unsigned char) * SPARSE_M01_N)

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
        PARAMS.lue[p] = params['LUE'][0][p]
        PARAMS.smrz0[p] = params['smrz0'][0][p]
        PARAMS.smrz1[p] = params['smrz1'][0][p]
        PARAMS.vpd0[p] = params['vpd0'][0][p]
        PARAMS.vpd1[p] = params['vpd1'][0][p]
        PARAMS.tmin0[p] = params['tmin0'][0][p]
        PARAMS.tmin1[p] = params['tmin1'][0][p]
        PARAMS.ft0[p] = params['ft0'][0][p]
        PARAMS.ft1[p] = params['ft1'][0][p]
        PARAMS.cue[p] = params['CUE'][0][p]

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
    date_fpar_ongoing = None
    for step in tqdm(range(num_steps)):
        date = date_start + datetime.timedelta(days = step)
        date_str = date.strftime('%Y%m%d')
        doy = int(date.strftime('%j'))
        year = int(date.year)

        # Read in soil moisture ("smsf" and "smrz") and soil temperature ("tsoil") data
        drivers = config['data']['drivers']
        read_flat_short((drivers['smsf'] % date_str).encode('UTF-8'), SPARSE_M09_N, smsf)
        read_flat_short((drivers['smrz0'] % date_str).encode('UTF-8'), SPARSE_M09_N, smrz0)
        read_flat((drivers['tsoil'] % date_str).encode('UTF-8'), SPARSE_M09_N, tsoil)
        # MERRA-2 daily variables
        read_flat((drivers['tmin'] % (year, date_str)).encode('UTF-8'), SPARSE_M09_N, tmin)
        read_flat((drivers['tsurf'] % (year, date_str)).encode('UTF-8'), SPARSE_M09_N, tsurf)
        read_flat((drivers['qv2m'] % (year, date_str)).encode('UTF-8'), SPARSE_M09_N, qv2m)
        read_flat((drivers['t2m'] % (year, date_str)).encode('UTF-8'), SPARSE_M09_N, t2m)
        read_flat((drivers['ps'] % (year, date_str)).encode('UTF-8'), SPARSE_M09_N, ps)
        read_flat((drivers['swrad'] % (year, date_str)).encode('UTF-8'), SPARSE_M09_N, swrad)

        # Read in fPAR data; to do so, we need to first find the nearest
        #   8-day composite date
        p = int(date.strftime('%j'))
        # Find the date of the fPAR data associated with this 8-day span
        if date.year % 4 == 0:
            date_fpar = PERIODS_DATES_LEAP[bisect_right(PERIODS, p) - 1]
        else:
            date_fpar = PERIODS_DATES[bisect_right(PERIODS, p) - 1]

        # Only read-in the fPAR data (and deflate it) once, for every
        #   8-day period
        if date_fpar_ongoing is None or date_fpar_ongoing != date_fpar:
            date_fpar_ongoing = date_fpar # The currently used fPAR data date
            check_fpar_qc = 1 # Assume we're checking fPAR QC flags
            # These have to be allocated differently for use with low-level functions
            in_bytes = size_in_bytes(DFNT_UINT8) * NCOL1KM * NROW1KM
            h5_fpar0     = <unsigned char*>calloc(sizeof(unsigned char), <size_t>in_bytes)
            h5_fpar_qc   = <unsigned char*>calloc(sizeof(unsigned char), <size_t>in_bytes)
            h5_fpar_clim = <unsigned char*>calloc(sizeof(unsigned char), <size_t>in_bytes)
            # Load the corresponding fPAR data (as a NumPy array)
            fpar_filename = config['data']['drivers']['fpar'] % (
                str(date.year) + date_fpar.strftime('%m%d'))
            # Byte-string versions of fPAR filenames
            fpar_filename_bs = fpar_filename.encode('UTF-8')
            fpar_clim_filename_bs = config['data']['fpar_clim'].encode('UTF-8')

            # Load and deflate the fPAR climatology...
            # The climatology field names are difficult; first, we need to get
            #   the ordinal of this 8-day period, counting starting from 1
            period = list(PERIODS).index(int(date_fpar.strftime("%j"))) + 1
            # Then, we need to format the string, e.g.:
            #   "fpar_clim_M01_day089_per12"
            clim_field = f'fpar_clim_M01_day{date_fpar.strftime("%j")}'\
                f'_per{str(period).zfill(2)}'
            read_hdf5(
                fpar_clim_filename_bs, clim_field.encode('utf-8'),
                H5T_STD_U8LE, h5_fpar_clim)
            deflate(h5_fpar_clim, fpar_clim, DFNT_UINT8, 'M01'.encode('UTF-8'))

            # Read and deflate the fPAR data and QC flags
            if os.path.exists(fpar_filename):
                read_hdf5(fpar_filename_bs, 'fpar_M01', H5T_STD_U8LE, h5_fpar0)
                read_hdf5(fpar_filename_bs, 'fpar_qc_M01', H5T_STD_U8LE, h5_fpar_qc)
                deflate(h5_fpar0, fpar0, DFNT_UINT8, 'M01'.encode('UTF-8'))
                deflate(h5_fpar_qc, fpar_qc, DFNT_UINT8, 'M01'.encode('UTF-8'))
            # If the fPAR data are not available for a given date (e.g., prior
            #   to Feburary 2000), use the climatology only
            else:
                print(f'No fPAR file for date {date_fpar.strftime("%j")} -- Using fPAR climatology')
                check_fpar_qc = 0
                fpar0 = fpar_clim

            free(h5_fpar0)
            free(h5_fpar_qc)
            free(h5_fpar_clim)

        # Option to schedule the rate at which litterfall enters SOC pools
        if config['model']['litterfall']['scheduled']:
            # Get the file covering the 8-day period in which this DOY falls
            filename = config['data']['litterfall_schedule']\
                % str(periods[doy-1]).zfill(2)
            read_flat(filename.encode('UTF-8'), SPARSE_M01_N, litter_rate)

        # Iterate over each 9-km pixel
        for i in prange(SPARSE_M09_N, nogil = True):
            par[i] = photosynth_active_radiation(swrad[i])
            vpd[i] = vapor_pressure_deficit(qv2m[i], ps[i], t2m[i])
            # SMRZ is in parts-per-thousand, convert to wetness (%)
            smrz[i] = rescale_smrz(
                smrz0[i] / 10.0, 100.0 * SMRZ_MIN[i], 100.0 * SMRZ_MAX[i])

            # Iterate over each nested 1-km pixel
            for j in prange(M01_NESTED_IN_M09):
                # Hence, (i) indexes the 9-km pixel and k the 1-km pixel
                k = (M01_NESTED_IN_M09 * i) + j
                pft = PFT[k]
                # Make sure to fill output grids with the FILL_VALUE,
                #   otherwise they may contain zero (0) at invalid data
                w_mult[k] = FILL_VALUE
                t_mult[k] = FILL_VALUE
                e_mult[k] = FILL_VALUE
                ft[k] = FILL_VALUE
                f_tmin[k] = FILL_VALUE
                f_vpd[k] = FILL_VALUE
                f_smrz[k] = FILL_VALUE
                rh_total[k] = FILL_VALUE
                soc_total[k] = FILL_VALUE
                nee[k] = FILL_VALUE
                gpp[k] = FILL_VALUE
                npp[k] = FILL_VALUE
                if is_valid(pft, tsoil[i], LITTERFALL[k]) == 0:
                    continue # Skip invalid PFTs

                # If surface soil temp. is above freezing, mark Thawed (=1),
                #   otherwise Frozen (=0)
                ft[k] = PARAMS.ft0[pft]
                # NOTE: TS/Tsurf is stored in deg K in MERRA-2
                if tsurf[i] >= 273.15:
                    ft[k] = PARAMS.ft1[pft]

                # Compute the environmental constraints
                f_tmin[k] = linear_constraint(
                    tmin[i], PARAMS.tmin0[pft], PARAMS.tmin1[pft], 0)
                f_vpd[k] = linear_constraint(
                    vpd[i], PARAMS.vpd0[pft], PARAMS.vpd1[pft], 1)
                f_smrz[k] = linear_constraint(
                    smrz[i], PARAMS.smrz0[pft], PARAMS.smrz1[pft], 0)
                e_mult[k] = ft[k] * f_tmin[k] * f_vpd[k] * f_smrz[k]

                # Determine the value of fPAR based on QC flag;
                #   bad pixels have either:
                #   1 in the left-most bit (SCF_QC bit = "Pixel not produced at all")
                fpar = <float>fpar0[k] # Otherwise, we're good
                if check_fpar_qc == 1:
                    if bits_from_uint32(7, 7, fpar_qc[k]) == 1:
                        fpar = <float>fpar_clim[k]
                    #   Or, anything other than 00 ("Clear") in bits 3-4
                    elif bits_from_uint32(3, 4, fpar_qc[k]) > 0:
                        fpar = <float>fpar_clim[k]
                # Then, check that we're not out of range
                if fpar0[k] > 100 and fpar_clim[k] <= 100:
                    fpar = <float>fpar_clim[k]
                elif fpar0[k] > 100:
                    continue # Skip this pixel
                fpar = fpar / 100.0 # Convert from [0,100] to [0,1]
                if DEBUG == 1:
                    fpar_final[k] = fpar

                # Compute GPP and NPP
                gpp[k] = fpar * par[i] * e_mult[k] * PARAMS.lue[pft]
                gpp[k] = fmax(0, gpp[k]) # Guard against negative values
                npp[k] = gpp[k] * PARAMS.cue[pft]

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
                # Compute NEE
                reco = rh_total[k] + (gpp[k] * (1 - PARAMS.cue[pft]))
                nee[k] = reco - gpp[k]

        # Optionally create restart files for each C pool, at the beginning
        #   of a new year
        if config['model']['restart']['create_file']:
            if date.month == 1 and date.day == 1:
                filename = (config['model']['restart']['output_file']\
                    % (date_str, 0)).encode('UTF-8')
                write_flat(filename, SPARSE_M01_N, SOC0)
                filename = (config['model']['restart']['output_file']\
                    % (date_str, 1)).encode('UTF-8')
                write_flat(filename, SPARSE_M01_N, SOC1)
                filename = (config['model']['restart']['output_file']\
                    % (date_str, 2)).encode('UTF-8')
                write_flat(filename, SPARSE_M01_N, SOC2)

        # If averaging from 1-km to 9-km resolution is requested...
        fmt = config['model']['output_format']
        suffix = '%s_%s' % (date_str, fmt) # e.g., "*_YYYYMMDD_M09land_*"
        suffix = suffix.encode('UTF-8')
        output_fields = list(map(
            lambda x: x.upper(), config['model']['output_fields']))

        output_dir = config['model']['output_dir']
        output_type = config['model']['output_type'].upper()
        fid = 0
        if fmt == 'M01land':
            out_fname_tpl = '%s/L4Cython_%%s_%s_%s.flt32' % (output_dir, date, fmt)
            if 'GPP' in output_fields:
                to_numpy(gpp, SPARSE_M01_N).tofile(out_fname_tpl % 'GPP')
            if 'NPP' in output_fields:
                to_numpy(npp, SPARSE_M01_N).tofile(out_fname_tpl % 'NPP')
            if 'EMULT' in output_fields:
                to_numpy(e_mult, SPARSE_M01_N).tofile(out_fname_tpl % 'Emult')
            if 'RH' in output_fields:
                to_numpy(rh_total, SPARSE_M01_N).tofile(out_fname_tpl % 'RH')
            if 'NEE' in output_fields:
                to_numpy(nee, SPARSE_M01_N).tofile(out_fname_tpl % 'NEE')
            if 'TMULT' in output_fields:
                to_numpy(t_mult, SPARSE_M01_N).tofile(out_fname_tpl % 'Tmult')
            if 'WMULT' in output_fields:
                to_numpy(w_mult, SPARSE_M01_N).tofile(out_fname_tpl % 'Wmult')
            if 'SOC' in output_fields:
                to_numpy(soc_total, SPARSE_M01_N).tofile(out_fname_tpl % 'SOC')
        elif fmt in ('M09', 'M09land'):
            inflated = 1 if fmt == 'M09' else 0
            output_func = write_resampled
        elif output_type == 'HDF5':
            inflated = 1
            output_func = write_fullres

        if 'GPP' in output_fields:
            fid = output_func(config, gpp, suffix, 'GPP', inflated, fid)
        if 'NPP' in output_fields:
            fid = output_func(config, npp, suffix, 'NPP', inflated, fid)
        if 'EMULT' in output_fields:
            fid = output_func(config, e_mult, suffix, 'Emult', inflated, fid)
        if 'RH' in output_fields:
            fid = output_func(config, rh_total, suffix, 'RH', inflated, fid)
        if 'NEE' in output_fields:
            fid = output_func(config, nee, suffix, 'NEE', inflated, fid)
        if 'TMULT' in output_fields:
            fid = output_func(config, t_mult, suffix, 'Tmult', inflated, fid)
        if 'WMULT' in output_fields:
            fid = output_func(config, w_mult, suffix, 'Wmult', inflated, fid)
        if 'SOC' in output_fields:
            fid = output_func(config, soc_total, suffix, 'SOC', inflated, fid)

    PyMem_Free(PFT)
    PyMem_Free(LITTERFALL)
    PyMem_Free(SOC0)
    PyMem_Free(SOC1)
    PyMem_Free(SOC2)
    PyMem_Free(SMRZ_MAX)
    PyMem_Free(SMRZ_MIN)
    PyMem_Free(smsf)
    PyMem_Free(tsoil)
    PyMem_Free(smrz0)
    PyMem_Free(smrz)
    PyMem_Free(swrad)
    PyMem_Free(t2m)
    PyMem_Free(tmin)
    PyMem_Free(qv2m)
    PyMem_Free(ps)
    PyMem_Free(tsurf)
    PyMem_Free(vpd)
    PyMem_Free(par)
    PyMem_Free(litter_rate)
    PyMem_Free(rh0)
    PyMem_Free(rh1)
    PyMem_Free(rh2)
    PyMem_Free(rh_total)
    PyMem_Free(nee)
    PyMem_Free(w_mult)
    PyMem_Free(t_mult)
    PyMem_Free(soc_total)
    PyMem_Free(ft)
    PyMem_Free(e_mult)
    PyMem_Free(gpp)
    PyMem_Free(npp)
    PyMem_Free(f_tmin)
    PyMem_Free(f_vpd)
    PyMem_Free(f_smrz)


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
    # Load ancillary files: min and max root-zone soil moisture (SMRZ)
    read_flat(config['data']['smrz_min'].encode('UTF-8'), SPARSE_M09_N, SMRZ_MIN)
    read_flat(config['data']['smrz_max'].encode('UTF-8'), SPARSE_M09_N, SMRZ_MAX)


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
