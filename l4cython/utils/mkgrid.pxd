# cython: language_level=3

from libc.stdlib cimport calloc
from l4cython.utils.fixtures import SPARSE_M09_N, NCOL9KM, NROW9KM, NCOL1KM, NROW1KM

cdef inline unsigned char* deflate(unsigned char* grid_array, unsigned short data_type, bytes grid):
    # NOTE: The flat_array and grid_array are handled as uint8 regardless of
    #   what the actual data type is; it just works this way in spland.c
    cdef:
        spland_ref_struct lookup
        unsigned char* flat_array

    # Assume 9-km grid, this also helps avoid warnings when compiling
    in_bytes = sizeof(float) * NCOL9KM * NROW9KM
    out_bytes = sizeof(float) * SPARSE_M09_N
    if grid.decode('UTF-8') == 'M01':
        in_bytes = sizeof(float) * NCOL1KM * NROW1KM
        out_bytes = sizeof(float) * SPARSE_M09_N * 81

    flat_array = <unsigned char*>calloc(sizeof(unsigned char), <size_t>out_bytes)
    # NOTE: Using 9-km row/col for both 9-km and 1-km nested grids
    lookup.row = <unsigned short*>calloc(sizeof(unsigned short), SPARSE_M09_N)
    lookup.col = <unsigned short*>calloc(sizeof(unsigned short), SPARSE_M09_N)

    # Load the index lookup file
    spland_load_9km_rc(&lookup)

    # Inflate the output array
    if grid.decode('UTF-8') == 'M09':
        spland_deflate_9km(lookup, &grid_array, &flat_array, data_type)
    elif grid.decode('UTF-8') == 'M01':
        spland_deflate_1km(lookup, &grid_array, &flat_array, data_type)
    return flat_array


cdef inline unsigned char* inflate(unsigned char* flat_array, unsigned short data_type, bytes grid):
    '''
    The inflated array can be written to an output file using, e.g.:

        out_bytes = sizeof(float) * number_of_pixels
        fwrite(grid_array, sizeof(unsigned char), <size_t>out_bytes, fid)

    Parameters
    ----------
    flat_array : unsigned char*
        The flattened (1D or "sparse land") array
    data_type : unsigned short
        The numeric code representing the data type
    grid : bytes
        The pixel size of the gridded data, e.g., "M09" for 9-km data or
        "M01" for 1-km data
    '''
    # NOTE: The flat_array and grid_array are handled as uint8 regardless of
    #   what the actual data type is; it just works this way in spland.c
    cdef:
        spland_ref_struct lookup
        unsigned char* grid_array

    # Assume 9-km grid, this also helps avoid warnings when compiling
    in_bytes = sizeof(float) * SPARSE_M09_N
    out_bytes = sizeof(float) * NCOL9KM * NROW9KM
    if grid.decode('UTF-8') == 'M01':
        in_bytes = sizeof(float) * SPARSE_M09_N * 81
        out_bytes = sizeof(float) * NCOL1KM * NROW1KM

    grid_array = <unsigned char*>calloc(sizeof(unsigned char), <size_t>out_bytes)
    # NOTE: Using 9-km row/col for both 9-km and 1-km nested grids
    lookup.row = <unsigned short*>calloc(sizeof(unsigned short), SPARSE_M09_N)
    lookup.col = <unsigned short*>calloc(sizeof(unsigned short), SPARSE_M09_N)

    # Load the index lookup file
    spland_load_9km_rc(&lookup)

    # Inflate the output array
    if grid.decode('UTF-8') == 'M09':
        spland_inflate_init_9km(&grid_array, data_type)
        spland_inflate_9km(lookup, &flat_array, &grid_array, data_type)
    elif grid.decode('UTF-8') == 'M01':
        spland_inflate_init_1km(&grid_array, data_type)
        spland_inflate_1km(lookup, &flat_array, &grid_array, data_type)
    return grid_array


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
    void spland_inflate_init_1km(void* dest_p, const unsigned int dataType)
    void set_fillval_UUTA(void* vDest_p, const unsigned int dataType, const size_t atSlot)
