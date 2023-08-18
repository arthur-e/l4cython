# cython: language_level=3
# distutils: sources = ["src/spland.c"]
# distutils: include_dirs = ["src/"]

'''
Tools for manipulating grid_array or flat_array EASE-Grid 2.0 array data.
'''

import cython
import tempfile
import numpy as np
from libc.stdlib cimport free, calloc
from libc.stdio cimport fopen, fread, fclose, fwrite
from l4cython.utils cimport open_fid
from l4cython.utils.fixtures import SPARSE_M09_N, SPARSE_M01_N, NCOL9KM, NROW9KM, NCOL1KM, NROW1KM, DFNT_FLOAT32, DFNT_FLOAT64, DFNT_UINT8, DFNT_INT8, DFNT_UINT16, DFNT_INT16, DFNT_UINT32, DFNT_INT32, READ, WRITE
from l4cython.utils.spland cimport spland_ref_struct, spland_inflate_9km, spland_inflate_init_9km, spland_inflate_1km, spland_inflate_init_1km, spland_load_9km_rc
# Implicit importing of inflate() function from mkgrid.pxd

@cython.boundscheck(False)
@cython.wraparound(False)
def deflate_file(filename, grid = 'M09'):
    '''
    Converts a gridded (2D), binary file to a deflated (1D or "sparse land")
    representation of the global EASE-Grid 2.0. The output file is written to
    the same directory as the input file.

    Parameters
    ----------
    filename : str
        The input filename, which should contain a substring like "M09"
        or "M01"
    grid : str
        The pixel size of the gridded data, e.g., "M09" for 9-km data or
        "M01" for 1-km data
    '''
    # NOTE: The flat_array and grid_array are handled as uint8 regardless of
    #   what the actual data type is; it just works this way in spland.c
    cdef:
        unsigned char* grid_array

    # Assume 9-km grid, this also helps avoid warnings when compiling
    in_bytes = sizeof(float) * NCOL9KM * NROW9KM
    out_bytes = sizeof(float) * SPARSE_M09_N
    if grid == 'M01':
        in_bytes = sizeof(float) * NCOL1KM * NROW1KM
        out_bytes = sizeof(float) * SPARSE_M09_N * 81
    grid_array = <unsigned char*>calloc(sizeof(unsigned char), <size_t>in_bytes)

    # Convert the unicode filename to a C string
    filename_byte_string = filename.encode('UTF-8')
    cdef char* fname = filename_byte_string

    # Infer data type by file extension, e.g., *.flt32, *.int32, etc.
    ext = filename_byte_string.decode('UTF-8').split('.').pop()
    data_type = DFNT_FLOAT32
    if ext == 'int32':
        data_type = DFNT_INT32
    elif ext == 'uint16':
        data_type = DFNT_UINT16

    # Read in the inflated array
    fid = fopen(fname, 'rb')
    if fid == NULL:
        print('ERROR -- File not found: %s' % filename_byte_string.decode('UTF-8'))
    fread(grid_array, sizeof(unsigned char), <size_t>in_bytes, fid)
    fclose(fid)

    # Inflate the output array
    grid = grid.encode('UTF-8')
    cdef char* c_grid = grid
    flat_array = deflate(grid_array, data_type, c_grid)

    output_filename = filename_byte_string\
        .decode('UTF-8')\
        .replace('M01', 'M01land')\
        .replace('M09', 'M09land')\
        .encode('UTF-8')
    cdef char* ofname = output_filename

    # Write the output file
    fid = fopen(ofname, 'wb')
    fwrite(flat_array, sizeof(unsigned char), <size_t>out_bytes, fid)
    fclose(fid)
    free(flat_array)
    free(grid_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def inflate_file(filename, grid = 'M09'):
    '''
    Converts a flat (1D or "land" format), binary file to an inflated (2D)
    representation of the global EASE-Grid 2.0. The output file is written to
    the same directory as the input file.

    Note that the inflation code in spland.c can mess up NoData values; values
    like -9999 become much larger (more negative). This may be because it
    handles all data as uint8 (unsigned char). I have no interest in debugging
    spland.c as it was written by someone else.

    Parameters
    ----------
    filename : str
        The input filename, which should contain a substring like "M09land"
        or "M01land"
    grid : str
        The pixel size of the gridded data, e.g., "M09" for 9-km data or
        "M01" for 1-km data
    '''
    # NOTE: The flat_array and grid_array are handled as uint8 regardless of
    #   what the actual data type is; it just works this way in spland.c
    cdef:
        unsigned char* flat_array

    # Assume 9-km grid, this also helps avoid warnings when compiling
    in_bytes = sizeof(float) * SPARSE_M09_N
    out_bytes = sizeof(float) * NCOL9KM * NROW9KM
    if grid == 'M01':
        in_bytes = sizeof(float) * SPARSE_M09_N * 81
        out_bytes = sizeof(float) * NCOL1KM * NROW1KM
    flat_array = <unsigned char*>calloc(sizeof(unsigned char), <size_t>in_bytes)

    # Convert the unicode filename to a C string
    filename_byte_string = filename.encode('UTF-8')
    cdef char* fname = filename_byte_string

    # Infer data type by file extension, e.g., *.flt32, *.int32, etc.
    ext = filename_byte_string.decode('UTF-8').split('.').pop()
    data_type = DFNT_FLOAT32
    if ext == 'int32':
        data_type = DFNT_INT32
    elif ext == 'uint16':
        data_type = DFNT_UINT16

    # Read in the deflated array
    fid = fopen(fname, 'rb')
    if fid == NULL:
        print('ERROR -- File not found: %s' % filename_byte_string.decode('UTF-8'))
    fread(flat_array, sizeof(unsigned char), <size_t>in_bytes, fid)
    fclose(fid)

    # Inflate the output array
    grid = grid.encode('UTF-8')
    cdef char* c_grid = grid
    grid_array = inflate(flat_array, data_type, c_grid)

    output_filename = filename_byte_string\
        .decode('UTF-8')\
        .replace('M01land', 'M01')\
        .replace('M09land', 'M09')\
        .encode('UTF-8')
    cdef char* ofname = output_filename

    # Write the output file
    fid = fopen(ofname, 'wb')
    fwrite(grid_array, sizeof(unsigned char), <size_t>out_bytes, fid)
    fclose(fid)
    free(flat_array)
    free(grid_array)


def write_deflated(
        output_filename, grid_numpy_array, data_type = DFNT_FLOAT32):
    '''
    Given a gridded (2D) array as a NumPy array, deflates the array to the
    flat (1D or "sparse land") format and writes to a file. This function
    works by writing the NumPy array to a temporary file, then having C's
    low-level `fread()` read back the array as bytes. Then, we can properly
    deflate the file and write to disk. This is necessary because `deflate()`
    only works for C arrays.

    For now, only NumPy arrays with 1-km elements should be passed.

    Parameters
    ----------
    output_filename : str
    grid_numpy_array : numpy.ndarray
    data_type : int
        Defaults to `DFNT_FLOAT32`
    '''
    cdef:
        char* fname
        char* ofname
        unsigned char* grid_array
        unsigned char* deflated_array
    if data_type == DFNT_FLOAT32:
        sz = sizeof(float)
    elif data_type == DFNT_FLOAT64:
        sz = sizeof(double)
    elif data_type == DFNT_INT8:
        sz = sizeof(char)
    elif data_type == DFNT_UINT8:
        sz = sizeof(unsigned char)
    elif data_type == DFNT_INT16:
        sz = sizeof(signed short)
    elif data_type == DFNT_UINT16:
        sz = sizeof(unsigned short)
    elif data_type == DFNT_INT32:
        sz = sizeof(signed int)
    elif data_type == DFNT_UINT32:
        sz = sizeof(unsigned int)
    in_bytes = sz * NCOL1KM * NROW1KM
    out_bytes = sz * SPARSE_M01_N
    grid_array = <unsigned char*>calloc(sizeof(unsigned char), <size_t>in_bytes)
    deflated_array = <unsigned char*>calloc(sizeof(unsigned char), <size_t>out_bytes)

    tmp = tempfile.NamedTemporaryFile()
    grid_numpy_array.tofile(tmp.name) # Write to memory
    # Get the filename as C bytes, then open for reading
    tmp_filename = tmp.name.encode('UTF-8')
    fname = tmp_filename
    fid = open_fid(fname, READ)
    if fid == NULL:
        print('ERROR -- Temporary file "%s" not readable' % tmp.name)
    fread(grid_array, sizeof(unsigned char), <size_t>in_bytes, fid)
    fclose(fid)

    # Inflate to a 2D grid, then write to file
    deflated_array = deflate(grid_array, data_type, 'M01'.encode('UTF-8'))
    ofname = output_filename
    fid = open_fid(ofname, WRITE)
    fwrite(deflated_array, sizeof(unsigned char), <size_t>out_bytes, fid)
    fclose(fid)
    free(grid_array)
    free(deflated_array)


def write_inflated(
        output_filename, flat_numpy_array, data_type = DFNT_FLOAT32):
    '''
    Given a flat (1D or "sparse land") array as a NumPy array, inflates the
    array to the global grid and writes to a file. This function works by
    writing the NumPy array to a temporary file, then having C's low-level
    `fread()` read back the array as bytes. Then, we can properly inflate the
    file and write to disk. This is necessary because `inflate()` only works
    for C arrays.

    For now, only NumPy arrays with 9-km elements should be passed; there is
    no support implemented for writing inflated 1-km arrays.

    NOTE: `output_filename` is expected as bytes, not a string; if starting
    with a string, use:

        write_inflated(filename.encode('UTF-8'), ...)

    Parameters
    ----------
    output_filename : bytes
    flat_numpy_array : numpy.ndarray
    data_type : int
        Defaults to `DFNT_FLOAT32`
    '''
    cdef:
        char* fname
        char* ofname
        unsigned char* flat_array
        unsigned char* inflated_array
    if data_type == DFNT_FLOAT32:
        sz = sizeof(float)
    elif data_type == DFNT_FLOAT64:
        sz = sizeof(double)
    elif data_type == DFNT_INT8:
        sz = sizeof(char)
    elif data_type == DFNT_UINT8:
        sz = sizeof(unsigned char)
    elif data_type == DFNT_INT16:
        sz = sizeof(signed short)
    elif data_type == DFNT_UINT16:
        sz = sizeof(unsigned short)
    elif data_type == DFNT_INT32:
        sz = sizeof(signed int)
    elif data_type == DFNT_UINT32:
        sz = sizeof(unsigned int)
    in_bytes = sz * NCOL1KM * NROW1KM
    out_bytes = sz * SPARSE_M01_N
    flat_array = <unsigned char*>calloc(sizeof(unsigned char), <size_t>in_bytes)
    inflated_array = <unsigned char*>calloc(sizeof(unsigned char), <size_t>out_bytes)

    tmp = tempfile.NamedTemporaryFile()
    flat_numpy_array.tofile(tmp.name) # Write to memory
    # Get the filename as C bytes, then open for reading
    tmp_filename = tmp.name.encode('UTF-8')
    fname = tmp_filename
    fid = open_fid(fname, READ)
    if fid == NULL:
        print('ERROR -- Temporary file "%s" not readable' % tmp.name)
    fread(flat_array, sizeof(unsigned char), <size_t>in_bytes, fid)
    fclose(fid)

    # Inflate to a 2D grid, then write to file
    inflated_array = inflate(
        flat_array, DFNT_FLOAT32, 'M09'.encode('UTF-8'))
    ofname = output_filename
    fid = open_fid(ofname, WRITE)
    fwrite(inflated_array, sizeof(unsigned char), <size_t>out_bytes, fid)
    fclose(fid)
    free(flat_array)
    free(inflated_array)
