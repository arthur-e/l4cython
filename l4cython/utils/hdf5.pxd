# cython: language_level=3

cdef extern from "src/hdf5.h":
    ctypedef long long int hid_t # int64
    ctypedef int herr_t

    int H5P_DEFAULT
    int H5S_ALL
    hid_t H5T_IEEE_F32LE
    hid_t H5T_STD_U8LE

    # To open an HDF5 file
    hid_t H5Fopen(char* filename, unsigned flags, hid_t access_plist)
    # To open a dataset within an HDF5 file
    hid_t H5Dopen1(hid_t file_id, char* name)
    # To read the data from the open dataset
    herr_t H5Dread(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id,
        hid_t file_space_id, hid_t plist_id, void* buf)


cdef inline void read_hdf5(
        char* filename, char* field, hid_t dtype, void* buff):
    '''
    Parameters
    ----------
    filename : char*
        The filename of the HDF5 file, as a byte string
    field : char*
        The dataset name to read from, as a byte string
    dtype : hid_t
        The data type, e.g., H5T_IEEE_F32LE
    buff : void*
        A pointer to an array buffer to receive the data
    '''
    # NOTE: In order of arguments...
    #   0 should be equivalent to hex code 0x0000u ("H5F_ACC_RDONLY")
    #   0 is the definition of H5P_DEFAULT in H5Ppublic.h
    fid = H5Fopen(filename, 0, H5P_DEFAULT)
    dsetid = H5Dopen1(fid, field)
    ret = H5Dread(dsetid, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buff)
