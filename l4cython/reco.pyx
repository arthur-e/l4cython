# cython: language_level=3

# SMAP Level 4 Carbon (L4C) heterotrophic respiration calculation, based
#   on Version 6 state and parameters

import cython
import datetime
import numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from respiration cimport BPLUT, arrhenius, linear_constraint

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

# We leave rh_total as a NumPy array because it is one we want to
#   write to disk; this has to be a global variable so it will
#   receive heap allocation
OUT_M01 = np.full((SPARSE_M01_N,), np.nan, dtype = np.float32)

# Allocate memory for, and populate, the PFT map
cdef unsigned char* PFT
# Allocate memory for SOC and litterfall (NPP) files
# NOTE: While we should be using PyMem_Free later, these variables will
#   be in use for the life of the program, so we let them free up only when
#   the program exits; for some reason, a segfault is encountered when
#   using: PyMem_Free(SOC1)
cdef:
    float* SOC0
    float* SOC1
    float* SOC2
    float* NPP
SOC0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
SOC1 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
SOC2 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
NPP = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)

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
@cython.wraparound(False)
def main(int num_steps = 2177):
    '''
    Forward run of the L4C soil decomposition and heterotrophic respiration
    algorithm. Starts on March 31, 2015 and continues for the specified
    number of time steps.

    Parameters
    ----------
    num_steps : int
        Number of (daily) time steps to compute forward
    '''
    PFT = <unsigned char*> PyMem_Malloc(sizeof(unsigned char) * SPARSE_M01_N)
    for idx, data in enumerate(np.fromfile('%s/SMAP_L4C_PFT_map_M01land.uint8' % ANC_DATA_DIR, np.uint8)):
        PFT[idx] = data
    # Read in SOC and NPP data; this has to be done by element, it seems
    for idx, data in enumerate(np.fromfile(
            '%s/tcf_NRv91_C0_M01land_0002089.flt32' % SOC_DATA_DIR, np.float32)):
        SOC0[idx] = data
    for idx, data in enumerate(np.fromfile(
            '%s/tcf_NRv91_C1_M01land_0002089.flt32' % SOC_DATA_DIR, np.float32)):
        SOC1[idx] = data
    for idx, data in enumerate(np.fromfile(
            '%s/tcf_NRv91_C2_M01land_0002089.flt32' % SOC_DATA_DIR, np.float32)):
        SOC2[idx] = data
    for idx, data in enumerate(np.fromfile(
            '%s/tcf_NRv91_npp_sum_M01land.flt32' % SOC_DATA_DIR, np.float32) / 365):
        NPP[idx] = data
    cdef:
        Py_ssize_t i
        Py_ssize_t j
        Py_ssize_t k
        float* rh0
        float* rh1
        float* rh2
        float* rh_total
        float* w_mult
        float* t_mult
        float k_mult
    rh0 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    rh1 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    rh2 = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    rh_total = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    w_mult = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    t_mult = <float*> PyMem_Malloc(sizeof(float) * SPARSE_M01_N)
    date_start = datetime.datetime.strptime(ORIGIN, '%Y%m%d')
    for step in range(num_steps):
        date = date_start + datetime.timedelta(days = step)
        date = date.strftime('%Y%m%d')
        # Convert to percentage units
        smsf = 100 * np.fromfile(
            '%s/L4_SM_gph_Vv6032_smsf_M09land_%s.flt32' % (L4SM_DATA_DIR, date),
            dtype = np.float32)
        tsoil = np.fromfile(
            '%s/L4_SM_gph_Vv6032_tsoil_M09land_%s.flt32' % (L4SM_DATA_DIR, date),
            dtype = np.float32)
        # Iterate over each 9-km pixel
        for i in range(0, SPARSE_M09_N):
            # Iterate over each nested 1-km pixel
            for j in range(0, M01_NESTED_IN_M09):
                # Hence, (i) indexes the 9-km pixel and k the 1-km pixel
                k = (M01_NESTED_IN_M09 * i) + j
                pft = int(PFT[k])
                if pft not in (1, 2, 3, 4, 5, 6, 7, 8):
                    continue
                w_mult[k] = linear_constraint(
                    smsf[i], params.smsf0[pft], params.smsf1[pft], 0)
                t_mult[k] = arrhenius(tsoil[i], params.tsoil[pft], TSOIL1, TSOIL2)
                k_mult = w_mult[k] * t_mult[k]
                rh0[k] = k_mult * SOC0[k] * params.decay_rate[pft]
                rh1[k] = k_mult * SOC1[k] * params.decay_rate[pft] * KSTRUCT
                rh2[k] = k_mult * SOC2[k] * params.decay_rate[pft] * KRECAL
                # "the adjustment...to account for material transferred into the
                #   slow pool during humification" (Jones et al. 2017 TGARS, p.5)
                rh1[k] = rh1[k] * (1 - params.f_structural[pft])
                rh_total[k] = rh0[k] + rh1[k] + rh2[k]
                # Calculate change in SOC pools; NPP[i] is daily litterfall
                SOC0[k] = (NPP[k] * params.f_metabolic[pft]) - rh0[k]
                SOC1[k] = (NPP[k] * (1 - params.f_metabolic[pft])) - rh1[k]
                SOC2[k] = (params.f_structural[pft] * rh1[k]) - rh2[k]
        # TODO FIXME Implicit break
        OUT_M01 = to_numpy(rh_total, SPARSE_M01_N)
        OUT_M01.tofile('%s/L4Cython_RH_%s_M01land.flt32' % (OUTPUT_DIR, date))
        break


cdef to_numpy(float *ptr, int n):
    '''
    Converts a typed memoryview to a NumPy array.

    Parameters
    ----------
    ptr : float*
        A pointer to the typed memoryview
    n : int
        The number of array elements

    Returns
    -------
    numpy.ndarray
    '''
    cdef int i
    arr = np.full((n,), np.nan, dtype = np.float32)
    for i in range(n):
        arr[i] = ptr[i]
    return arr
