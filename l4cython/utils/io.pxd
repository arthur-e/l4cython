# cython: language_level=3

import numpy as np
from libc.stdio cimport FILE, fopen, fread, fclose, fwrite

cdef extern from "src/io.h":
    char* READ
    char* WRITE

cdef:
    FILE* fid

cdef inline FILE* open_fid(bytes filename_byte_string, bytes mode):
    '''
    Open a file using a filename given as a Python byte string.

    Parameters
    ----------
    filename_byte_string : bytes
        Should be a Python string encoded, e.g., `filename.encode("UTF-8")`
    mode : bytes

    Returns
    -------
    FILE*
    '''
    cdef char* fname = filename_byte_string # Convert to a C string
    cdef FILE* fid = fopen(fname, mode)
    if fid is NULL:
        print('ERROR -- File not found: %s' % filename_byte_string.decode('UTF-8'))
    return fid


cdef inline void read_flat(char* filename, int n_elem, float* arr):
    '''
    Reads in global, 9-km data from a flat file (*.flt32).

    Parameters
    ----------
    filename : char*
        The filename to read
    n_elem : int
        The number of array elements
    arr : float*
        The destination array buffer
    '''
    fid = open_fid(filename, READ)
    fread(arr, sizeof(float), <size_t>sizeof(float)*n_elem, fid)
    fclose(fid)


cdef inline void read_flat_char(char* filename, int n_elem, unsigned char* arr):
    '''
    Reads in global, 9-km data from a flat file (*.uint8).

    Parameters
    ----------
    filename : char*
        The filename to read
    n_elem : int
        The number of array elements
    arr : short int*
        The destination array buffer
    '''
    fid = open_fid(filename, READ)
    fread(arr, sizeof(unsigned char), <size_t>sizeof(unsigned char)*n_elem, fid)
    fclose(fid)


cdef inline void read_flat_short(char* filename, int n_elem, short int* arr):
    '''
    Reads in global, 9-km data from a flat file (*.int16).

    Parameters
    ----------
    filename : char*
        The filename to read
    n_elem : int
        The number of array elements
    arr : short int*
        The destination array buffer
    '''
    fid = open_fid(filename, READ)
    fread(arr, sizeof(short int), <size_t>sizeof(short int)*n_elem, fid)
    fclose(fid)


cdef inline to_numpy_char(unsigned char *ptr, int n):
    '''
    Converts a typed memoryview to a NumPy array.

    Parameters
    ----------
    ptr : unsigned char*
        A pointer to the typed memoryview
    n : int
        The number of array elements

    Returns
    -------
    numpy.ndarray
    '''
    cdef int i
    arr = np.full((n,), 255, dtype = np.uint8)
    for i in range(n):
        arr[i] = ptr[i]
    return arr


cdef inline to_numpy(float *ptr, int n):
    '''
    Converts a typed memoryview to a NumPy array.

    Parameters
    ----------
    ptr : float*
        A pointer to the typed memoryview
    n : int
        The number of array elements

    Returns
    -------
    numpy.ndarray
    '''
    cdef int i
    arr = np.full((n,), np.nan, dtype = np.float32)
    for i in range(n):
        arr[i] = ptr[i]
    return arr


cdef inline to_numpy_double(double *ptr, int n):
    '''
    Converts a typed memoryview to a NumPy array with "double" (64-bit)
    precision.

    Parameters
    ----------
    ptr : float*
        A pointer to the typed memoryview
    n : int
        The number of array elements

    Returns
    -------
    numpy.ndarray
    '''
    cdef int i
    arr = np.full((n,), np.nan, dtype = np.float64)
    for i in range(n):
        arr[i] = ptr[i]
    return arr


cdef inline void write_flat(char* filename, int n_elem, float* arr):
    '''
    Writes floating-point data to a flat file (*.flt32).

    Parameters
    ----------
    filename : char*
        The filename to read
    n_elem : int
        The number of array elements
    arr : float*
        The array buffer containing the data to write to file
    '''
    fid = open_fid(filename, WRITE)
    fwrite(arr, sizeof(float), <size_t>sizeof(float)*n_elem, fid)
    fclose(fid)
