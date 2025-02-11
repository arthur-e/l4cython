# cython: language_level=3

cdef extern from "src/dec2bin.c":
    unsigned long int bits_from_uint32(
        const signed long int start_at, const signed long int end_at,
        unsigned long int value) nogil
