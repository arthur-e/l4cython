import numpy as np

cdef extern from "src/spland.h":
    ctypedef struct spland_ref_struct:
        unsigned short* row # NOTE: 16-bit unsigned integer
        unsigned short* col

    int spland_load_9km_rc(spland_ref_struct* SPLAND)

    void spland_deflate_9km(spland_ref_struct SPLAND, void* src_p, void* dest_p, const unsigned int dataType)
    void spland_deflate_1km(spland_ref_struct SPLAND, void* src_p, void* dest_p, const unsigned int dataType)

    void spland_inflate_9km(spland_ref_struct SPLAND, void* src_p, void* dest_p, const unsigned int dataType)
    void spland_inflate_1km(spland_ref_struct SPLAND, void* src_p, void* dest_p, const unsigned int dataType)


cdef inline unsigned short M_2D_B0(unsigned short x, unsigned short y, unsigned short n_y):
    return ((x) * n_y) + y
