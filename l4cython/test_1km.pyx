# cython: language_level=3

# SMAP Level 4 Carbon (L4C) heterotrophic respiration calculation, based
#   on Version 6 state and parameters

import cython
import datetime
import numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

DEF M01_NESTED_IN_M09 = 9 * 9
# Number of grid cells in sparse ("land") arrays
DEF SPARSE_M09_N = 1664040
DEF SPARSE_M01_N = M01_NESTED_IN_M09 * SPARSE_M09_N
DEF ANC_DATA_DIR = '/anx_lagr3/arthur.endsley/SMAP_L4C/ancillary_data'
DEF SOC_DATA_DIR = '/anx_lagr4/SMAP/L4C_code/tcf/output/NRv91'
DEF L4SM_DATA_DIR = '/anx_lagr4/SMAP/L4SM/Vv6032'
DEF OUTPUT_DIR = '/anx_lagr3/arthur.endsley/SMAP_L4C/L4C_Science/Cython/v20220106'
DEF ORIGIN = '20150331' # First day, in YYYYMMDD

# Additional Tsoil parameter (fixed for all PFTs)
cdef float TSOIL1 = 66.02 # deg K
cdef float TSOIL2 = 227.13 # deg K
# Additional SOC decay parameters (fixed for all PFTs)
cdef float KSTRUCT = 0.4 # Muliplier *against* base decay rate
cdef float KRECAL = 0.0093

# Allocate memory for, and populate, the PFT map
cdef unsigned char* PFT
cdef int _i = 0
PFT = <unsigned char*> PyMem_Malloc(sizeof(unsigned char) * SPARSE_M09_N)
for data in np.fromfile('%s/SMAP_L4C_PFT_map_M09land.uint8' % ANC_DATA_DIR, np.uint8):
    PFT[_i] = data
    _i = _i + 1

# Allocate memory for SOC and litterfall (NPP) files
cdef:
    unsigned int* SOC0
    unsigned int* SOC1
    unsigned int* SOC2
    float* NPP
SOC0 = <unsigned int*> PyMem_Malloc(sizeof(unsigned int) * SPARSE_M01_N)
SOC1 = <unsigned int*> PyMem_Malloc(sizeof(unsigned int) * SPARSE_M01_N)
SOC2 = <unsigned int*> PyMem_Malloc(sizeof(unsigned int) * SPARSE_M01_N)
NPP = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)

# Read in SOC and NPP data; this has to be done by element, it seems
# for i, data in enumerate(np.fromfile(
#         '%s/tcf_NRv91_C0_M01land_0002089.flt32' % SOC_DATA_DIR, np.float32).astype(np.uint16)):
#     SOC0[i] = data
# for i, data in enumerate(np.fromfile(
#         '%s/tcf_NRv91_C1_M01land_0002089.flt32' % SOC_DATA_DIR, np.float32).astype(np.uint16)):
#     SOC1[i] = data
# for i, data in enumerate(np.fromfile(
#         '%s/tcf_NRv91_C2_M01land_0002089.flt32' % SOC_DATA_DIR, np.float32).astype(np.uint16)):
#     SOC2[i] = data
# for i, data in enumerate(np.fromfile(
#         '%s/tcf_NRv91_npp_sum_M01land.flt32' % SOC_DATA_DIR, np.float32) / 365):
#     NPP[i] = data

cdef struct BPLUT:
    float smsf0[9] # wetness [0-100%]
    float smsf1[9] # wetness [0-100%]
    float tsoil[9] # deg K
    float cue[9]
    float f_metabolic[9]
    float f_structural[9]
    float decay_rate[9]

# NOTE: Must have an (arbitrary) value in 0th position to avoid overflow of
#   indexing (as PFT=0 is not used and C starts counting at 0)
cdef BPLUT params
params.smsf0[:] = [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
params.smsf1[:] = [0, 25.0, 30.5, 39.8, 31.3, 44.9, 50.5, 25.0, 25.1]
params.tsoil[:] = [0, 266.05, 392.24, 233.94, 265.23, 240.71, 261.42, 253.98, 281.69]
params.cue[:] = [0, 0.687, 0.469, 0.755, 0.799, 0.649, 0.572, 0.708, 0.705]
params.f_metabolic[:] = [0, 0.49, 0.71, 0.67, 0.67, 0.62, 0.76, 0.78, 0.78]
params.f_structural[:] = [0, 0.3, 0.3, 0.7, 0.3, 0.35, 0.55, 0.5, 0.8]
params.decay_rate[:] = [0, 0.020, 0.022, 0.031, 0.028, 0.013, 0.022, 0.019, 0.031]


@cython.boundscheck(False)
def main(int num_steps = 2177):
    pass
