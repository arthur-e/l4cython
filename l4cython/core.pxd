# cython: language_level=3

cdef extern from "utils/src/spland.h":
    int N_PFT, FILL_VALUE
    int M01_NESTED_IN_M09, M03_NESTED_IN_M09
    int SPARSE_M09_N, SPARSE_M03_N, SPARSE_M01_N
    int NCOL1KM, NROW1KM, NCOL3KM, NROW3KM, NCOL9KM, NROW9KM
    int DFNT_FLOAT32, DFNT_FLOAT64, DFNT_UINT8, DFNT_INT8, DFNT_UINT16, DFNT_INT16, DFNT_UINT32, DFNT_INT32, DFNT_INT64


# Biome Properties Lookup Table (BPLUT)
cdef struct BPLUT:
    float lue[9] # Maximum light-use efficiency
    float smrz0[9] # Root-zone soil wetness [0-100%]
    float smrz1[9]
    float tmin0[9] # Minimum temperature (deg K)
    float tmin1[9]
    float vpd0[9] # Vapor pressure deficit (Pa)
    float vpd1[9]
    float ft0[9] # Multiplier when soil is (Frozen=0)
    float ft1[9] # Multiplier when soil is (Thawed=1)
    float smsf0[9] # Surface soil wetness [0-100%]
    float smsf1[9]
    float tsoil[9] # deg K
    float cue[9]
    float f_metabolic[9]
    float f_structural[9]
    float decay_rate[3][9]
