# cython: language_level=3
# distutils: sources = ["src/spland.c"]
# distutils: include_dirs = ["src/"]

'''
'''

import cython
import numpy as np
from libc.stdlib cimport free, calloc
from libc.stdio cimport fopen, fread, fclose, fwrite
from spland cimport spland_ref_struct, spland_inflate_9km, spland_inflate_init_9km, spland_load_9km_rc

DEF SPARSE_N = 1664040 # Number of grid cells in sparse ("land") arrays
DEF NCOL9KM = 3856
DEF NROW9KM = 1624

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
def main(filename):
    '''
    '''
    # Convert the unicode filename to a C string
    filename_byte_string = filename.encode('UTF-8')
    cdef char* fname = filename_byte_string

    # Load the index lookup file
    cdef spland_ref_struct lookup
    # Allocate space and read in row/col reference data
    # NOTE: Using 9-km row/col for both 9-km and 1-km nested grids
    lookup.row = <unsigned short*>calloc(sizeof(unsigned short), SPARSE_N);
    lookup.col = <unsigned short*>calloc(sizeof(unsigned short), SPARSE_N);
    spland_load_9km_rc(&lookup)

    # Read in the deflated array
    in_bytes = sizeof(float) * SPARSE_N
    deflated = <unsigned char*>calloc(sizeof(unsigned char), <size_t>in_bytes)
    fid = fopen(fname, 'rb')
    fread(deflated, sizeof(unsigned char), <size_t>in_bytes, fid)
    fclose(fid)

    # Inflate the output array
    out_bytes = sizeof(float) * NCOL9KM * NROW9KM
    inflated = <unsigned char*>calloc(sizeof(unsigned char), <size_t>out_bytes)
    spland_inflate_init_9km(&inflated, DFNT_FLOAT32);
    spland_inflate_9km(lookup, &deflated, &inflated, DFNT_FLOAT32)

    # Write the output file
    output_filename = filename_byte_string\
        .decode('UTF-8').replace('M09land', 'M09').encode('UTF-8')
    cdef char* ofname = output_filename
    fid = fopen(ofname, 'wb')
    fwrite(inflated, sizeof(unsigned char), <size_t>out_bytes, fid)
    fclose(fid)

    free(deflated)
    free(inflated)
