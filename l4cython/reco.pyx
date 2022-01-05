# cython: language_level=3

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
