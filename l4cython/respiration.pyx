# cython: language_level=3

# SMAP Level 4 Carbon (L4C) heterotrophic respiration calculation, based
#   on Version 6 state and parameters

import numpy as np

# Number of grid cells in sparse ("land") arrays
DEF SPARSE_N = 1664040

# Additional Tsoil parameter (fixed for all PFTs)
cdef float TSOIL1 = 66.02 # deg K
cdef float TSOIL2 = 227.13 # deg K
# Additional SOC decay parameters (fixed for all PFTs)
cdef float KSTRUCT = 0.4 # Muliplier *against* base decay rate
cdef float KRECAL = 0.0093

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

# The PFT map
cdef char PFT[SPARSE_N]
cdef:
    float SOC0[SPARSE_N]
    float SOC1[SPARSE_N]
    float SOC2[SPARSE_N]

PFT[:] = np.fromfile(
    '/anx_lagr3/arthur.endsley/SMAP_L4C/ancillary_data/SMAP_L4C_PFT_map_9km.int8', np.int8)
SOC0[:] = np.fromfile(
    '/anx_lagr3/arthur.endsley/SMAP_L4C/ancillary_data/tcf_natv91_C0_M09land_2015089.flt32', np.float32)
SOC1[:] = np.fromfile(
    '/anx_lagr3/arthur.endsley/SMAP_L4C/ancillary_data/tcf_natv91_C1_M09land_2015089.flt32', np.float32)
SOC2[:] = np.fromfile(
    '/anx_lagr3/arthur.endsley/SMAP_L4C/ancillary_data/tcf_natv91_C2_M09land_2015089.flt32', np.float32)
