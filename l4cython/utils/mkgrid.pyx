# cython: language_level=3
# distutils: sources = ["src/spland.c"]
# distutils: include_dirs = ["src/"]

'''
'''

import cython
import numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from tqdm import tqdm
from spland cimport spland_ref_struct, spland_inflate_9km, spland_load_9km_rc

# Number of grid cells in sparse ("land") arrays
DEF SPARSE_N = 1664040
DEF NCOL9KM = 3856
DEF NROW9KM = 1624
DEF LAND_R_FILE = '/anx_v2/laj/smap/code/landdomdef/output/MCD12Q1_M09land_row.uint16'
DEF LAND_C_FILE = '/anx_v2/laj/smap/code/landdomdef/output/MCD12Q1_M09land_col.uint16'

# Allocate memory for inflated arrays
cdef:
    float* INFLATED
INFLATED = <float*> PyMem_Malloc(sizeof(float) * NCOL9KM * NROW9KM)

# Row-column index lookups
cdef:
    float DEFLATED[SPARSE_N]
    unsigned short ROWS[SPARSE_N]
    unsigned short COLS[SPARSE_N]

@cython.boundscheck(False)
@cython.wraparound(False)
def main():
    '''
    '''
    DEFLATED[:] = np.fromfile('/anx_lagr3/arthur.endsley/SMAP_L4C/L4C_Science/Cython/v20230523/L4Cython_RH_20150331_M09land.flt32', np.float32)
    cdef spland_ref_struct lookup
    spland_load_9km_rc(&lookup)


def load_9km_rc():
    'Load the row-column index lookup'
    ROWS[:] = np.fromfile(LAND_R_FILE, np.uint16)
    COLS[:] = np.fromfile(LAND_C_FILE, np.uint16)
