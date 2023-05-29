# cython: language_level=3
# distutils: sources = ["src/spland.c"]
# distutils: include_dirs = ["src/"]

'''
'''

import cython
import numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.stdlib cimport calloc
from libc.stdio cimport fopen, fread, fclose
from tqdm import tqdm
from spland cimport spland_ref_struct, spland_inflate_9km, spland_inflate_init_9km, spland_load_9km_rc

DEF SPARSE_N = 1664040 # Number of grid cells in sparse ("land") arrays
DEF NCOL9KM = 3856
DEF NROW9KM = 1624
DEF LAND_R_FILE = '/anx_v2/laj/smap/code/landdomdef/output/MCD12Q1_M09land_row.uint16'
DEF LAND_C_FILE = '/anx_v2/laj/smap/code/landdomdef/output/MCD12Q1_M09land_col.uint16'

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

# TOTALLY UNECESSARY but getting a segfault without
cdef:
    float DUMMY[SPARSE_N]

@cython.boundscheck(False)
@cython.wraparound(False)
def main():
    '''
    '''
    # TOTALLY UNECESSARY but getting a segfault without
    DUMMY[:] = np.fromfile('/anx_lagr3/arthur.endsley/SMAP_L4C/L4C_Science/Cython/v20230523/L4Cython_RH_20150331_M09land.flt32', np.float32)

    # Load the index lookup file
    cdef spland_ref_struct lookup
    spland_load_9km_rc(&lookup)

    n_bytes = sizeof(float) * SPARSE_N
    DEFLATED = <unsigned char*>calloc(sizeof(unsigned char), <size_t>n_bytes)
    fid = fopen('/anx_lagr3/arthur.endsley/SMAP_L4C/L4C_Science/Cython/v20230523/L4Cython_RH_20150331_M09land.flt32', 'rb')
    fread(DEFLATED, sizeof(unsigned char), <size_t>n_bytes, fid)
    fclose(fid)

    n_bytes = sizeof(float) * NCOL9KM * NROW9KM
    INFLATED = <unsigned char*>calloc(sizeof(unsigned char), <size_t>n_bytes)
    spland_inflate_init_9km(&INFLATED, DFNT_FLOAT32);
    # spland_inflate_9km(lookup, &DEFLATED, &INFLATED, DFNT_FLOAT32)
