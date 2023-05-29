cdef extern from "src/spland.h":
    ctypedef struct spland_ref_struct:
        unsigned short* row # NOTE: 16-bit unsigned integer
        unsigned short* col

    int spland_load_9km_rc(spland_ref_struct* SPLAND)

    void spland_deflate_9km(spland_ref_struct SPLAND, void* src_p, void* dest_p, const unsigned int dataType)
    void spland_deflate_1km(spland_ref_struct SPLAND, void* src_p, void* dest_p, const unsigned int dataType)

    void spland_inflate_9km(spland_ref_struct SPLAND, void* src_p, void* dest_p, const unsigned int dataType)
    void spland_inflate_1km(spland_ref_struct SPLAND, void* src_p, void* dest_p, const unsigned int dataType)

    void spland_inflate_init_9km(void* dest_p, const unsigned int dataType)
    void set_fillval_UUTA(void* vDest_p, const unsigned int dataType, const size_t atSlot)
