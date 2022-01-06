# cython: language_level=3

# SMAP Level 4 Carbon (L4C) heterotrophic respiration calculation, based
#   on Version 6 state and parameters

# TODO: Re-write using typed memoryviews?

import datetime
import numpy as np

# Number of grid cells in sparse ("land") arrays
DEF SPARSE_N = 1664040
DEF ANC_DATA_DIR = '/anx_lagr3/arthur.endsley/SMAP_L4C/ancillary_data'
DEF L4SM_DATA_DIR = '/anx_lagr4/SMAP/L4SM/Vv6032'
DEF ORIGIN = '20150331' # First day, in YYYYMMDD format

# Additional Tsoil parameter (fixed for all PFTs)
cdef float TSOIL1 = 66.02 # deg K
cdef float TSOIL2 = 227.13 # deg K
# Additional SOC decay parameters (fixed for all PFTs)
cdef float KSTRUCT = 0.4 # Muliplier *against* base decay rate
cdef float KRECAL = 0.0093

# The PFT map
cdef char PFT[SPARSE_N]
cdef:
    float SMRZ_MIN[SPARSE_N]
    float SMRZ_MAX[SPARSE_N]
    float SOC0[SPARSE_N]
    float SOC1[SPARSE_N]
    float SOC2[SPARSE_N]
    float NPP[SPARSE_N]

PFT[:] = np.fromfile('%s/SMAP_L4C_PFT_map_9km.int8' % ANC_DATA_DIR, np.int8)
SMRZ_MIN[:] = np.fromfile(
    '%s/Natv91_daily_smrz_M09_min.flt32' % ANC_DATA_DIR, np.float32)
SMRZ_MAX[:] = np.fromfile(
    '%s/Natv91_daily_smrz_M09_max.flt32' % ANC_DATA_DIR, np.float32)
SOC0[:] = np.fromfile(
    '%s/tcf_natv91_C0_M09land_2015089.flt32' % ANC_DATA_DIR, np.float32)
SOC1[:] = np.fromfile(
    '%s/tcf_natv91_C1_M09land_2015089.flt32' % ANC_DATA_DIR, np.float32)
SOC2[:] = np.fromfile(
    '%s/tcf_natv91_C2_M09land_2015089.flt32' % ANC_DATA_DIR, np.float32)
NPP[:] = np.fromfile(
    '%s/tcf_natv91_npp_sum_M09land.flt32' % ANC_DATA_DIR, np.float32) / 365

cdef struct BPLUT:
    float smsf0[8] # wetness [0-100%]
    float smsf1[8] # wetness [0-100%]
    float tsoil[8] # deg K
    float cue[8]
    float f_metabolic[8]
    float f_structural[8]
    float decay_rate[8]

cdef BPLUT params
params.smsf0[:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
params.smsf1[:] = [25.0, 30.5, 39.8, 31.3, 44.9, 50.5, 25.0, 25.1]
params.tsoil[:] = [266.05, 392.24, 233.94, 265.23, 240.71, 261.42, 253.98, 281.69]
params.cue[:] = [0.687, 0.469, 0.755, 0.799, 0.649, 0.572, 0.708, 0.705]
params.f_metabolic[:] = [0.49, 0.71, 0.67, 0.67, 0.62, 0.76, 0.78, 0.78]
params.f_structural[:] = [0.3, 0.3, 0.7, 0.3, 0.35, 0.55, 0.5, 0.8]
params.decay_rate[:] = [0.020, 0.022, 0.031, 0.028, 0.013, 0.022, 0.019, 0.031]


def main(int num_steps = 2177):
    '''
    Forward run of the L4C soil decompositiona nd heterotrophic respiration
    algorithm. Starts on March 31, 2015 and continues for the specified
    number of time steps.

    Parameters
    ----------
    num_steps : int
        Number of (daily) time steps to compute forward
    '''
    cdef:
        Py_ssize_t i
        float rh[SPARSE_N]
        float smsf[SPARSE_N]
        float smrz[SPARSE_N]
        float tsoil[SPARSE_N]
    rh = np.zeroes((SPARSE_N,))
    date_start = datetime.datetime.strftime(ORIGIN, '%Y%m%d')
    for step in range(num_steps):
        date = date_start + datetime.timedelta(step)
        smsf[:] = np.fromfile(
            '%s/L4_SM_gph_Vv6032_smsf_M09land_%s.flt32' % (L4SM_DATA_DIR, date))
        smrz[:] = np.fromfile(
            '%s/L4_SM_gph_Vv6032_smrz_M09land_%s.flt32' % (L4SM_DATA_DIR, date))
        break


cdef float[:] rescale_smrz(float[:] smrz_array):
    cdef:
        Py_ssize_t i
        float[SPARSE_N] smrz_out
        float smrz0, smrz_min, srmz_max, smrz_norm
    for i in range(0, SPARSE_N):
        # Convert to percentage units
        smrz0 = 100 * smrz_array[i]
        smrz_min = 100 * SMRZ_MIN[i]
        smrz_max = 100 * SMRZ_MAX[i]
        # Clip input SMRZ to the lower, upper bounds
        if smrz0 < smrz_min:
            smrz0 = smrz_min
        elif smrz0 > smrz_max:
            smrz0 = smrz_max
        smrz_norm = 1 + (100 * ((smrz0 - smrz_min) / (smrz_max - smrz_min)))
        # Log-transform normalized data and rescale to range between
        #   5.0 and 100 ()% saturation)
        smrz_out[i] = 5 + (95 * (np.log(smrz_norm) / np.log(101)))
    return smrz_out
