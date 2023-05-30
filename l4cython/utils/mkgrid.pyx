# cython: language_level=3
# distutils: sources = ["src/spland.c"]
# distutils: include_dirs = ["src/"]

'''
Tools for manipulating inflated or deflated EASE-Grid 2.0 array data.
'''

import cython
import numpy as np
from libc.stdlib cimport free, calloc
from libc.stdio cimport fopen, fread, fclose, fwrite
from spland cimport spland_ref_struct, spland_inflate_9km, spland_inflate_init_9km, spland_inflate_1km, spland_inflate_init_1km, spland_load_9km_rc

DEF SPARSE_N = 1664040 # Number of grid cells in sparse ("land") arrays
DEF NCOL9KM = 3856
DEF NROW9KM = 1624
DEF NCOL1KM = 34704
DEF NROW1KM = 14616

# From hntdefs.h
DEF DFNT_FLOAT32 = 5
DEF DFNT_FLOAT64 = 6
DEF DFNT_INT8   = 20
DEF DFNT_UINT8  = 21
DEF DFNT_INT16  = 22
DEF DFNT_UINT16 = 23
DEF DFNT_INT32  = 24
DEF DFNT_UINT32 = 25
DEF DFNT_INT64  = 26
DEF DFNT_UINT64 = 27


@cython.boundscheck(False)
@cython.wraparound(False)
def inflate(filename, grid = 'M09'):
    '''
    Converts a flat (1D or "land" format), binary file to an inflated (2D)
    representation on the global EASE-Grid 2.0. The output file is written to
    the same directory as the input file.

    Parameters
    ----------
    filename : str
        The input filename, which should contain a substring like "M09land"
        or "M01land"
    grid : str
        The pixel size of the gridded data, e.g., "M09" for 9-km data or
        "M01" for 1-km data
    '''
    cdef:
        spland_ref_struct lookup
        unsigned char* deflated
        unsigned char* inflated
        
    # Assume 9-km grid, this also helps avoid warnings when compiling
    in_bytes = sizeof(float) * SPARSE_N
    out_bytes = sizeof(float) * NCOL9KM * NROW9KM
    if grid == 'M01':
        in_bytes = sizeof(float) * SPARSE_N * 81
        out_bytes = sizeof(float) * NCOL1KM * NROW1KM

    deflated = <unsigned char*>calloc(sizeof(unsigned char), <size_t>in_bytes)
    inflated = <unsigned char*>calloc(sizeof(unsigned char), <size_t>out_bytes)
    # NOTE: Using 9-km row/col for both 9-km and 1-km nested grids
    lookup.row = <unsigned short*>calloc(sizeof(unsigned short), SPARSE_N);
    lookup.col = <unsigned short*>calloc(sizeof(unsigned short), SPARSE_N);

    # Convert the unicode filename to a C string
    filename_byte_string = filename.encode('UTF-8')
    cdef char* fname = filename_byte_string

    # Infer data type by file extension, e.g., *.flt32, *.int32, etc.
    ext = filename_byte_string.decode('UTF-8').split('.').pop()
    data_type = DFNT_FLOAT32
    if ext == 'int32':
        data_type = DFNT_INT32
    elif ext == 'uint16':
        data_type = DFNT_UINT16

    # Load the index lookup file
    spland_load_9km_rc(&lookup)

    # Read in the deflated array
    fid = fopen(fname, 'rb')
    if fid == NULL:
        print('ERROR -- File not found: %s' % filename_byte_string.decode('UTF-8'))
    fread(deflated, sizeof(unsigned char), <size_t>in_bytes, fid)
    fclose(fid)

    # Inflate the output array
    if grid == 'M09':
        spland_inflate_init_9km(&inflated, data_type);
        spland_inflate_9km(lookup, &deflated, &inflated, data_type)
    elif grid == 'M01':
        spland_inflate_init_1km(&inflated, data_type);
        spland_inflate_1km(lookup, &deflated, &inflated, data_type)

    output_filename = filename_byte_string\
        .decode('UTF-8')\
        .replace('M01land', 'M01')\
        .replace('M09land', 'M09')\
        .encode('UTF-8')
    cdef char* ofname = output_filename

    # Write the output file
    fid = fopen(ofname, 'wb')
    fwrite(inflated, sizeof(unsigned char), <size_t>out_bytes, fid)
    fclose(fid)

    free(deflated)
    free(inflated)
