# cython: language_level=3

cdef extern from "src/hdf5.h":
    ctypedef long long int hid_t # int64
    ctypedef int herr_t

    int H5P_DEFAULT
    int H5S_ALL
    int H5T_NATIVE_FLOAT

    # To open an HDF5 file
    hid_t H5Fopen(char* filename, unsigned flags, hid_t access_plist)
    # To open a dataset within an HDF5 file
    hid_t H5Dopen1(hid_t file_id, char* name)
    # To read the data from the open dataset
    herr_t H5Dread(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id,
        hid_t file_space_id, hid_t plist_id, void* buf)
