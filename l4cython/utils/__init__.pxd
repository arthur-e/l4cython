import numpy as np
from libc.stdio cimport FILE, fopen, fread, fclose, fwrite

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
