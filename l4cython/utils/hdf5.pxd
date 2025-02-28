# cython: language_level=3

cdef extern from "stdio.h":
    int remove(char* filename)

cdef extern from "src/hdf5.h":
    ctypedef long long int hid_t # int64
    ctypedef unsigned long long int hsize_t # uint6
    ctypedef int herr_t
    ctypedef int htri_t

    int H5P_DEFAULT # Default property list
    int H5S_ALL
    int H5F_ACC_TRUNC # Flag to "truncate" the file, "if it already exists, erasing all data previously stored in the file"
    int H5F_ACC_EXCL # Flag to fail on creating a file that already exists
    int H5S_SIMPLE # An n-dimensional data space
    hid_t H5T_IEEE_F32LE
    hid_t H5T_NATIVE_UINT8
    hid_t H5T_STD_U8LE

    # To open an HDF5 file
    hid_t H5Fopen(char* filename, unsigned flags, hid_t access_plist)
    # To open a dataset within an HDF5 file
    hid_t H5Dopen1(hid_t file_id, char* name)
    # To read the data from the open dataset
    herr_t H5Dread(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id,
        hid_t file_space_id, hid_t plist_id, void* buf)
    herr_t H5Dclose(hid_t dset_id)
    htri_t H5Fis_hdf5(char* filename) # Is it an HDF5 file?

    # To close or delete an HDF5 file
    herr_t H5Fclose(hid_t file_id)

    # To write an HDF5 file
    hid_t H5Fcreate(
        char* filename, unsigned char flags, hid_t fcpl_id, hid_t fapl_id)
    herr_t H5Dset_extent(hid_t dset_id, hsize_t* size)

    # To create a "data space"
    hid_t H5Screate_simple(int rank, hsize_t dims[], hsize_t maxdims[])

    # To create a group
    hid_t H5Gcreate(hid_t loc_id, char* name, hid_t lcpl_id, hid_t gcpl_id, hid_t gapl_id)

    # To create an HDF5 dataset
    hid_t H5Dcreate(
        hid_t loc_id, char* name, hid_t type_id, hid_t space_id, hid_t lcpl_id, hid_t dcpl_id, hid_t dapl_id)

    # To write data into an HDF5 dataset from a buffer
    herr_t H5Dwrite(
        hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, void* buff)


cdef inline hid_t create_1d_space(int nelem):
    '''
    Parameters
    ----------
    nelem : int
        The number of elements

    Returns
    -------
    hid_t
        The data space ID
    '''
    cdef hsize_t dims[1]
    cdef hsize_t max_dims[1]
    dims[:] = [nelem]
    max_dims[:] = [nelem]
    return H5Screate_simple(1, dims, max_dims)


cdef inline hid_t create_2d_space(int nrow, int ncol):
    '''
    Parameters
    ----------
    nrow : int
        The number of rows
    ncol : int
        The number of columns

    Returns
    -------
    hid_t
        The data space ID
    '''
    cdef hsize_t dims[2]
    cdef hsize_t max_dims[2]
    dims[:] = [nrow, ncol]
    max_dims[:] = [nrow, ncol]
    return H5Screate_simple(2, dims, max_dims)


cdef inline void close_hdf5(hid_t fid):
    '''
    Will only fail if the file already exists; will not overwrite.

    Parameters
    ----------
    fid : hid_t
        The file ID
    '''
    H5Fclose(fid)


cdef inline hid_t open_hdf5(char* filename):
    '''
    Will only fail if the file already exists; will not overwrite.

    Parameters
    ----------
    filename : char*
        The filename of the HDF5 file, as a byte string

    Returns
    -------
    hid_t
        The file ID
    '''
    if H5Fis_hdf5(filename) >= 0:
        remove(filename) # Delete the file
    # Using H5P_DEFAULT for args fcpl_id, fapl_id
    return H5Fcreate(filename, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT)


# Based on frm_hdf5_ReadData()
cdef inline void read_hdf5(
        char* filename, char* field, hid_t dtype, void* buff):
    '''
    Opens an HDF5 file and reads a given dataset.

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
    dset_id = H5Dopen1(fid, field)
    ret = H5Dread(dset_id, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buff)
    H5Dclose(dset_id)
    H5Fclose(fid)


# Based (loosely) on frm_hdf5_WriteData()
cdef inline void write_hdf5_dataset(
        hid_t fid, char* field, hid_t dtype, hid_t dspace, void* buff):
    '''
    Writes a dataset into an (already open) HDF5 file.

    Parameters
    ----------
    fid : hid_t
        The file ID
    field : char*
        The dataset name to write to, as a byte string
    dtype : hid_t
        The data type, e.g., H5T_IEEE_F32LE
    dspace : hid_t
    buff : void*
        A pointer to an array buffer from which to read the data
    '''
    cdef hid_t dest, gid
    # If there is no intermediate group, destination is the file
    dset_name = field.decode('UTF-8')
    dest = fid
    # In case of an intermediate group, i.e., "group/dataset_name"
    if '/' in dset_name:
        group, dset_name = field.decode('UTF-8').split('/')
        gid = H5Gcreate(
            fid, group.encode('UTF-8'), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)
        dest = gid

    dset_id = H5Dcreate( # Using H5P_DEFAULT for lcpl_id, dcpl_id, dapl_id
        dest, dset_name.encode('UTF-8'), dtype, dspace,
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)
    H5Dwrite(dset_id, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buff)
    H5Dclose(dset_id)
