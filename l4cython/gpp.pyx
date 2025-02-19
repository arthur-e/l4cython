# cython: language_level=3
# distutils: sources = ["utils/src/spland.c", "utils/src/uuta.c"]

'''
Assumptions:

- The fPAR dataset (an HDF5 file) has a field "fpar_M01" that contains the
    fPAR data and a field "fpar_qc_M01" that contains the QC flags.
'''

import cython
import datetime
import yaml
import numpy as np
import h5py
from bisect import bisect_right
from tempfile import NamedTemporaryFile
from libc.stdlib cimport calloc, free
from libc.stdio cimport FILE, fopen, fread, fclose, fwrite
from libc.math cimport fmax
from cython.parallel import prange
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from l4cython.constraints cimport linear_constraint
from l4cython.science cimport rescale_smrz, vapor_pressure_deficit, photosynth_active_radiation
from l4cython.utils cimport BPLUT, open_fid, to_numpy, to_numpy_char # FIXME
from l4cython.utils.mkgrid import write_numpy_inflated, write_numpy_deflated, deflate_file
from l4cython.utils.mkgrid cimport deflate, size_in_bytes
from l4cython.utils.hdf5 cimport read_hdf5, H5T_STD_U8LE, H5T_IEEE_F32LE
from l4cython.utils.dec2bin cimport bits_from_uint32
from l4cython.utils.fixtures import READ, DFNT_UINT8, DFNT_FLOAT32, NCOL1KM, NROW1KM, NCOL9KM, NROW9KM, N_PFT, load_parameters_table
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
    unsigned char* PFT
    unsigned char* PFT_MASK
    float* SMRZ_MAX
    float* SMRZ_MIN
PFT = <unsigned char*> PyMem_Malloc(sizeof(unsigned char) * SPARSE_M01_N)
SMRZ_MAX = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
SMRZ_MIN = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)

# Python arrays that want heap allocations must be global; this one is reused
#   for any array that needs to be written to disk (using NumPy)
OUT_M01 = np.full((SPARSE_M01_N,), np.nan, dtype = np.float32)
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
        int DEBUG
        float fpar

    smrz0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    smrz  = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    swrad = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    t2m   = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    tmin  = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    qv2m  = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    ps    = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    tsurf = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    vpd   = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
    par   = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M09_N)
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

    # These have to be allocated differently for use with low-level functions
    in_bytes = size_in_bytes(DFNT_UINT8) * NCOL1KM * NROW1KM
    h5_fpar0     = <unsigned char*>calloc(sizeof(unsigned char), <size_t>in_bytes)
    h5_fpar_qc   = <unsigned char*>calloc(sizeof(unsigned char), <size_t>in_bytes)
    h5_fpar_clim = <unsigned char*>calloc(sizeof(unsigned char), <size_t>in_bytes)

    # Read in configuration file, then load the global state variables
    if config is None:
        config = '../data/L4Cython_GPP_M01_config.yaml'
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

    # Read in the model parameters (Biome Properties Lookup Table, BPLUT)
    params = load_parameters_table(config['BPLUT'].encode('UTF-8'))
    for p in range(1, N_PFT + 1):
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

    # Begin forward time stepping
    date_fpar_ongoing = None
    for step in tqdm(range(num_steps)):
        date = date_start + datetime.timedelta(days = step)
        date_str = date.strftime('%Y%m%d')
        doy = int(date.strftime('%j'))

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
            # Load the corresponding fPAR data (as a NumPy array)
            fpar_filename = config['data']['drivers']['fpar'] % (
                str(date.year) + date_fpar.strftime('%m%d'))
            fpar_filename_bs = fpar_filename.encode('UTF-8')
            # Load and deflate the fPAR climatology
            fpar_clim_filename_bs = config['data']['fpar_clim'].encode('UTF-8')
            # Read and deflate the fPAR data and QC flags
            read_hdf5(fpar_filename_bs, 'fpar_M01', H5T_STD_U8LE, h5_fpar0)
            read_hdf5(fpar_filename_bs, 'fpar_qc_M01', H5T_STD_U8LE, h5_fpar_qc)
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
            fpar0 = deflate(h5_fpar0, DFNT_UINT8, 'M01'.encode('UTF-8'))
            fpar_qc = deflate(h5_fpar_qc, DFNT_UINT8, 'M01'.encode('UTF-8'))
            fpar_clim = deflate(h5_fpar_clim, DFNT_UINT8, 'M01'.encode('UTF-8'))

        # Read in the remaining surface meteorlogical data
        # We re-use a single NamedTemporaryFile(), overwriting its contents
        #   without creating a new instance; there should be little risk in
        #  this because each array is the same size
        tmp = NamedTemporaryFile()
        tmp_fname_bs = tmp.name.encode('UTF-8')
        # NOTE: Alternatively, try reading with read_hdf5, then copying the
        #   array (element-wise) into an unsigned char* buffer
        # fname_bs = (config['data']['drivers']['other'] % date_str).encode('UTF-8')
        # read_hdf5(fname_bs, 'QV2M_M09_AVG', H5T_IEEE_F32LE, qv2m)
        with h5py.File(
                config['data']['drivers']['other'] % date_str, 'r') as hdf:
            # SWRAD
            write_numpy_deflated(tmp_fname_bs, hdf['RADIATION_SHORTWAVE_DOWNWARD_FLUX'][:])
            read_flat(tmp_fname_bs, SPARSE_M09_N, swrad)
            # T2M_M09_MIN (tmin)
            write_numpy_deflated(tmp_fname_bs, hdf['T2M_M09_MIN'][:])
            read_flat(tmp_fname_bs, SPARSE_M09_N, tmin)
            # T2M_M09_AVG (t2m)
            write_numpy_deflated(tmp_fname_bs, hdf['T2M_M09_AVG'][:])
            read_flat(tmp_fname_bs, SPARSE_M09_N, t2m)
            # QV2M
            write_numpy_deflated(tmp_fname_bs, hdf['QV2M_M09_AVG'][:])
            read_flat(tmp_fname_bs, SPARSE_M09_N, qv2m)
            # PS
            write_numpy_deflated(tmp_fname_bs, hdf['SURFACE_PRESSURE'][:])
            read_flat(tmp_fname_bs, SPARSE_M09_N, ps)
            # SMRZ (before rescaling)
            write_numpy_deflated(tmp_fname_bs, hdf['SM_ROOTZONE_WETNESS'][:])
            read_flat(tmp_fname_bs, SPARSE_M09_N, smrz0)
            # NOTE: TS/Tsurf is stored in deg C, to great annoyance
            # TS/Tsurf
            write_numpy_deflated(tmp_fname_bs, hdf['TS_M09_DEGC_AVG'][:])
            read_flat(tmp_fname_bs, SPARSE_M09_N, tsurf)

        # Iterate over 9-km grid
        for i in prange(SPARSE_M09_N, nogil = True):
            par[i] = photosynth_active_radiation(swrad[i])
            vpd[i] = vapor_pressure_deficit(qv2m[i], ps[i], t2m[i])
            smrz[i] = rescale_smrz(smrz0[i], SMRZ_MIN[i], SMRZ_MAX[i])

            # TODO See if it is actually faster to do this in a single thread
            #   i.e., with range(); could be overhead assoc. with small task
            # Iterate over each nested 1-km pixel
            for j in prange(M01_NESTED_IN_M09):
                # Hence, (i) indexes the 9-km pixel and k the 1-km pixel
                k = (M01_NESTED_IN_M09 * i) + j
                pft = PFT[k]
                # Make sure to fill output grids with the FILL_VALUE,
                #   otherwise they may contain zero (0) at invalid data
                e_mult[k] = FILL_VALUE
                ft[k] = FILL_VALUE
                f_tmin[k] = FILL_VALUE
                f_vpd[k] = FILL_VALUE
                f_smrz[k] = FILL_VALUE
                gpp[k] = FILL_VALUE
                npp[k] = FILL_VALUE
                if is_valid(pft) == 0:
                    continue # Skip invalid PFTs
                # If surface soil temp. is above freezing, mark Thawed (=1),
                #   otherwise Frozen (=0)
                ft[k] = PARAMS.ft0[pft]
                # NOTE: TS/Tsurf is stored in deg C, to great annoyance
                if tsurf[i] >= 0:
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
                if bits_from_uint32(7, 7, fpar_qc[k]) == 1:
                    fpar = <float>fpar_clim[k]
                #   Or, anything other than 00 ("Clear") in bits 3-4
                elif bits_from_uint32(3, 4, fpar_qc[k]) > 0:
                    fpar = <float>fpar_clim[k]
                else:
                    fpar = <float>fpar0[k] # Otherwise, we're good
                # Then, check that we're not out of range
                if fpar0[k] > 100 and fpar_clim[k] <= 100:
                    fpar = <float>fpar_clim[k]
                elif fpar0[k] > 100:
                    continue # Skip this pixel
                fpar = fpar / 100.0 # Convert from [0,100] to [0,1]
                if DEBUG == 1:
                    fpar_final[k] = fpar

                # Finally, compute GPP and NPP
                gpp[k] = fpar * par[i] * e_mult[k] * PARAMS.lue[pft]
                npp[k] = gpp[k] * PARAMS.cue[pft]


        # If averaging from 1-km to 9-km resolution is requested...
        fmt = config['model']['output_format']
        out_dir = config['model']['output_dir']
        output_fname_tpl = (
            '%s/L4Cython_%%s_%s_%s.flt32' % (out_dir, date_str, fmt))
        output_fields = list(map(
            lambda x: x.upper(), config['model']['output_fields']))
        if 'F_TMIN' in output_fields:
            output_filename = (output_fname_tpl % 'fTmin').encode('UTF-8')
            write_numpy_inflated(output_filename, to_numpy(f_tmin, SPARSE_M09_N))
        if 'F_VPD' in output_fields:
            output_filename = (output_fname_tpl % 'fVPD').encode('UTF-8')
            write_numpy_inflated(output_filename, to_numpy(f_vpd, SPARSE_M09_N))
        if 'F_SMRZ' in output_fields:
            output_filename = (output_fname_tpl % 'fSMRZ').encode('UTF-8')
            write_numpy_inflated(output_filename, to_numpy(f_smrz, SPARSE_M09_N))
        if 'F_FT' in output_fields:
            output_filename = (output_fname_tpl % 'fFT').encode('UTF-8')
            write_numpy_inflated(output_filename, to_numpy(ft, SPARSE_M09_N))
        if fmt in ('M09', 'M09land'):
            inflated = 1 if fmt == 'M09' else 0
            if 'GPP' in output_fields:
                output_filename = (output_fname_tpl % 'GPP').encode('UTF-8')
                write_resampled(output_filename, gpp, inflated)
            if 'NPP' in output_fields:
                output_filename = (output_fname_tpl % 'NPP').encode('UTF-8')
                write_resampled(output_filename, npp, inflated)
            if 'EMULT' in output_fields:
                output_filename = (output_fname_tpl % 'Emult').encode('UTF-8')
                write_resampled(output_filename, e_mult, inflated)
            if DEBUG == 1:
                output_filename = (output_fname_tpl % 'fPAR').encode('UTF-8')
                write_resampled(output_filename, fpar_final, inflated)
        else:
            if 'GPP' in output_fields:
                OUT_M01 = to_numpy(gpp, SPARSE_M01_N)
                OUT_M01.tofile(output_fname_tpl % 'GPP')
            if 'NPP' in output_fields:
                OUT_M01 = to_numpy(npp, SPARSE_M01_N)
                OUT_M01.tofile(output_fname_tpl % 'NPP')
            if 'EMULT' in output_fields:
                OUT_M01 = to_numpy(e_mult, SPARSE_M01_N)
                OUT_M01.tofile(output_fname_tpl % 'Emult')


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
    PyMem_Free(ft)
    PyMem_Free(e_mult)
    PyMem_Free(gpp)
    PyMem_Free(npp)
    PyMem_Free(f_tmin)
    PyMem_Free(f_vpd)
    PyMem_Free(f_smrz)
    PyMem_Free(fpar0)
    PyMem_Free(fpar_qc)
    PyMem_Free(fpar_clim)
    free(h5_fpar0)
    free(h5_fpar_qc)
    free(h5_fpar_clim)


def load_state(config):
    '''
    Populates global state variables with data.

    Parameters
    ----------
    config : dict
        The configuration data dictionary
    '''
    in_bytes = size_in_bytes(DFNT_UINT8) * NCOL9KM * NROW9KM
    h5_pft_mask = <unsigned char*>calloc(sizeof(unsigned char), <size_t>in_bytes)
    # Allocate space, read in 1-km PFT map
    ancillary_file_bs = config['data']['ancillary']['file'].encode('UTF-8')
    pft_map_field = config['data']['ancillary']['PFT_map'].encode('UTF-8')
    read_hdf5(ancillary_file_bs, pft_map_field, H5T_STD_U8LE, PFT)
    # Load ancillary files: min and max root-zone soil moisture (SMRZ);
    #   we write these NumPy arrays to deflated flat files in memory,
    #   then read them back in
    with h5py.File(config['data']['restart']['file'], 'r') as hdf:
        smrz_max = hdf[config['data']['restart']['smrz_max']][:]
        smrz_min = hdf[config['data']['restart']['smrz_min']][:]
    with NamedTemporaryFile() as tmp:
        write_numpy_deflated(tmp.name.encode('UTF-8'), smrz_max)
        read_flat(tmp.name.encode('UTF-8'), SPARSE_M09_N, SMRZ_MAX)
    with NamedTemporaryFile() as tmp:
        write_numpy_deflated(tmp.name.encode('UTF-8'), smrz_min)
        read_flat(tmp.name.encode('UTF-8'), SPARSE_M09_N, SMRZ_MIN)
    free(h5_pft_mask)


def mod15a2h_qc_fail(x):
    '''
    Returns pass/fail for QC flags based on the L4C fPAR QC protocol for the
    `FparLai_QC` band: Bad pixels have either `1` in the first bit ("Pixel not
    produced at all") or anything other than `00` ("clear") in bits 3-4.
    Output array is True wherever the array fails QC criteria. Compare to:

        np.vectorize(lambda v: v[0] == 1 or v[3:5] != '00')

    Parameters
    ----------
    x : numpy.ndarray
        Array where the last axis enumerates the unpacked bits
        (ones and zeros)

    Returns
    -------
    numpy.ndarray
        Boolean array with True wherever QC criteria are failed
    '''
    y = np.unpackbits(x[...,None], axis = 1)[...,-8:]
    # Emit 1 = FAIL if these two bits are not == "00"
    c1 = y[...,3:5].sum(axis = -1).astype(np.uint8)
    # Emit 1 = FAIL if 1st bit == 1 ("Pixel not produced at all")
    c2 = y[...,0]
    # Intermediate arrays are 1 = FAIL, 0 = PASS
    return (c1 + c2) > 0


cdef inline char is_valid(char pft) nogil:
    '''
    Checks to see if a given pixel is valid, based on the PFT.
    Modeled after `tcfModUtil_isInCell()` in the TCF code.

    Parameters
    ----------
    pft : char
        The Plant Functional Type (PFT)

    Returns
    -------
    char
        A value of 0 indicates the pixel is invalid, otherwise returns 1
    '''
    cdef char valid = 1 # Assume it's a valid pixel
    if pft not in (1, 2, 3, 4, 5, 6, 7, 8):
        valid = 0
    return valid


cdef inline void read_flat(char* filename, int n_elem, float* arr):
    '''
    Reads in global, 9-km data from a flat file (*.flt32).

    Parameters
    ----------
    filename : char*
        The filename to read
    n_elem : int
        The number of array elements
    arr : float*
        The destination array buffer
    '''
    fid = open_fid(filename, READ)
    fread(arr, sizeof(float), <size_t>sizeof(float)*n_elem, fid)
    fclose(fid)


cdef void write_resampled(
        bytes output_filename, float* array_data, int inflated = 1):
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
    # Write a flat (1D) file or inflate the file and then write
    if inflated == 0:
        data_resampled.tofile(output_filename.decode('UTF-8'))
    else:
        write_numpy_inflated(output_filename, data_resampled, grid = 'M09')
