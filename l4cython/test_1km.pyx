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
PFT = <unsigned char*> PyMem_Malloc(sizeof(unsigned char) * SPARSE_M01_N)
for i, data in enumerate(np.fromfile('%s/SMAP_L4C_PFT_map_M01land.uint8' % ANC_DATA_DIR, np.uint8)):
    PFT[i] = data

# Allocate memory for SOC and litterfall (NPP) files
# NOTE: While we should be using PyMem_Free later, these variables will
#   be in use for the life of the program, so we let them free up only when
#   the program exits; for some reason, a segfault is encountered when
#   using: PyMem_Free(SOC1)
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
for i, data in enumerate(np.fromfile(
        '%s/tcf_NRv91_C0_M01land_0002089.flt32' % SOC_DATA_DIR, np.float32).astype(np.uint16)):
    SOC0[i] = data
for i, data in enumerate(np.fromfile(
        '%s/tcf_NRv91_C1_M01land_0002089.flt32' % SOC_DATA_DIR, np.float32).astype(np.uint16)):
    SOC1[i] = data
for i, data in enumerate(np.fromfile(
        '%s/tcf_NRv91_C2_M01land_0002089.flt32' % SOC_DATA_DIR, np.float32).astype(np.uint16)):
    SOC2[i] = data
for i, data in enumerate(np.fromfile(
        '%s/tcf_NRv91_npp_sum_M01land.flt32' % SOC_DATA_DIR, np.float32) / 365):
    NPP[i] = data

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
    '''
    Forward run of the L4C soil decomposition and heterotrophic respiration
    algorithm. Starts on March 31, 2015 and continues for the specified
    number of time steps.

    Parameters
    ----------
    num_steps : int
        Number of (daily) time steps to compute forward
    '''
    cdef:
        Py_ssize_t i
        float rh0[SPARSE_M01_N]
        float rh1[SPARSE_M01_N]
        float rh2[SPARSE_M01_N]
        float w_mult[SPARSE_M01_N]
        float t_mult[SPARSE_M01_N]
        float k_mult[SPARSE_M01_N]
    # We leave rh_total as a NumPy array because it is one we want to
    #   write to disk
    rh_total = np.full((SPARSE_M01_N,), np.nan, dtype = np.float32)
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
                # Hence, (i) indexes the 9-km pixel and (i+j) the 1-km pixel
                pft = int(PFT[i+j])
                if pft not in (1, 2, 3, 4, 5, 6, 7, 8):
                    continue
                w_mult[i+j] = linear_constraint(
                    smsf[i], params.smsf0[pft], params.smsf1[pft], 0)
                t_mult[i+j] = arrhenius(tsoil[i], params.tsoil[pft], TSOIL1, TSOIL2)
                k_mult[i+j] = w_mult[i+j] * t_mult[i+j]
                rh0[i+j] = k_mult[i+j] * SOC0[i+j] * params.decay_rate[pft]
                rh1[i+j] = k_mult[i+j] * SOC1[i+j] * params.decay_rate[pft] * KSTRUCT
                rh2[i+j] = k_mult[i+j] * SOC2[i+j] * params.decay_rate[pft] * KRECAL
                # "the adjustment...to account for material transferred into the
                #   slow pool during humification" (Jones et al. 2017 TGARS, p.5)
                rh1[i+j] = rh1[i+j] * (1 - params.f_structural[pft])
                rh_total[i+j] = rh0[i+j] + rh1[i+j] + rh2[i+j]
        break # TODO FIXME


cdef float arrhenius(
        float tsoil, float beta0, float beta1, float beta2):
    '''
    The Arrhenius equation for response of enzymes to (soil) temperature,
    constrained to lie on the closed interval [0, 1].

    Parameters
    ----------
    tsoil : float
        Array of soil temperature in degrees K
    beta0 : float
        Coefficient for soil temperature (deg K)
    beta1 : float
        Coefficient for ... (deg K)
    beta2 : float
        Coefficient for ... (deg K)
    '''
    cdef float a, b, y0
    a = (1.0 / beta1)
    b = 1 / (tsoil - beta2)
    # This is the simple answer, but it takes on values >1
    y0 = np.exp(beta0 * (a - b))
    # Constrain the output to the interval [0, 1]
    if y0 > 1:
        return 1
    elif y0 < 0:
        return 0
    else:
        return y0


cdef float linear_constraint(
        float x, float xmin, float xmax, int reversed):
    '''
    Returns a linear ramp function, for deriving a value on [0, 1] from
    an input value `x`:

        if x >= xmax:
            return 1
        if x <= xmin:
            return 0
        return (x - xmin) / (xmax - xmin)

    Parameters
    ----------
    x: float
    xmin : float
        Lower bound of the linear ramp function
    xmax : float
        Upper bound of the linear ramp function
    reversed : int
        Type of ramp function: 1 for "reversed," i.e., function decreases
        as x increases

    Returns
    -------
    float
    '''
    assert reversed == 0 or reversed == 1
    if reversed == 1:
        if x >= xmax:
            return 0
        elif x < xmin:
            return 1
        else:
            return 1 - ((x - xmin) / (xmax - xmin))
    # For normal case
    if x >= xmax:
        return 1
    elif x < xmin:
        return 0
    else:
        return (x - xmin) / (xmax - xmin)
