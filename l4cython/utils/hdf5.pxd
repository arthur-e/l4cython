# cython: language_level=3

cdef extern from "stdio.h":
    int remove(char* filename)

cdef extern from "unistd.h":
    int F_OK
    int access(char* pathname, int how)

cdef extern from "src/hdf5.h":
    ctypedef long long int hid_t # int64
    ctypedef unsigned long long int hsize_t # uint6
    ctypedef int herr_t
    ctypedef int htri_t
    ctypedef herr_t H5E_auto1_t
    ctypedef herr_t H5E_auto2_t
    ctypedef enum H5D_layout_t:
        H5D_LAYOUT_ERROR
        H5D_COMPACT
        H5D_CONTIGUOUS
        H5D_CHUNKED
        H5D_VIRTUAL
        H5D_NLAYOUTS
    ctypedef enum H5FD_mpio_collective_opt_t:
        H5FD_MPIO_COLLECTIVE_IO
        H5FD_MPIO_INDIVIDUAL_IO
    ctypedef enum H5FD_mpio_xfer_t:
        H5FD_MPIO_INDEPENDENT
        H5FD_MPIO_COLLECTIVE
    ctypedef int MPI_Comm
    ctypedef int MPI_Info

    # Defined in mpi.h
    # https://support.hdfgroup.org/documentation/hdf5/latest/_intro_par_h_d_f5.html
    int MPI_COMM_WORLD
    int MPI_INFO_NULL
    int MPI_Init(int* argc, char** argv[])

    int H5E_DEFAULT # Default error stack
    int H5P_DEFAULT # Default property list
    int H5P_DATASET_CREATE # Default property list for dataset creation
    int H5P_FILE_CREATE # Properties for file creation
    int H5P_FILE_ACCESS # Properties for file access
    int H5P_FILE_ACCESS_DEFAULT # Properties for file access
    int H5P_DATASET_XFER # Dataset transfer property list class
    int H5S_ALL
    int H5F_ACC_TRUNC # Flag to "truncate" the file, "if it already exists, erasing all data previously stored in the file"
    int H5F_ACC_EXCL # Flag to fail on creating a file that already exists
    int H5S_SIMPLE # An n-dimensional data space
    int H5I_INVALID_HID # Is -1 for error return
    hid_t H5T_IEEE_F32LE
    hid_t H5T_NATIVE_UINT8
    hid_t H5T_STD_U8LE

    # For error handling
    herr_t H5Eset_auto2(hid_t estack_id, H5E_auto2_t func, void* client_data)

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

    # To create an HDF5 file
    hid_t H5Fcreate(
        char* filename, unsigned char flags, hid_t fcpl_id, hid_t fapl_id)

    # To create a "data space"
    hid_t H5Screate_simple(int rank, hsize_t dims[], hsize_t maxdims[])

    # To create or open a group
    hid_t H5Gcreate(hid_t loc_id, char* name, hid_t lcpl_id, hid_t gcpl_id, hid_t gapl_id)
    htri_t H5Lexists(hid_t loc_id, char* name, hid_t lapl_id)

    # To create an HDF5 dataset
    hid_t H5Dcreate(
        hid_t loc_id, char* name, hid_t type_id, hid_t space_id, hid_t lcpl_id, hid_t dcpl_id, hid_t dapl_id)

    # To write data into an HDF5 dataset from a buffer
    herr_t H5Dwrite(
        hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t dxpl_id, void* buff)

    # For property lists and compression
    hid_t H5Pcreate(hid_t cls_id)
    hid_t H5Pcopy(hid_t plist_id)
    hid_t H5Pclose(hid_t plist_id)
    herr_t H5Pset_dxpl_mpio(hid_t dxpl_id, H5FD_mpio_xfer_t xfer_mode)
    herr_t H5Pset_layout(hid_t plist_id, H5D_layout_t layout)
    herr_t H5Pset_chunk(hid_t plist_id, int ndims, hsize_t dim[])
    herr_t H5Pset_fill_value(hid_t plist_id, hid_t type_id, void* value)
    herr_t H5Pset_deflate(hid_t plist_id, unsigned char level)

    herr_t H5Pset_fapl_mpio(hid_t fapl_id, MPI_Comm comm, MPI_Info info)


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
    If the file already exists, it is deleted. It is up to external code
    to remember the file ID of a file if you want to keep working with it.

    Parameters
    ----------
    filename : char*
        The filename of the HDF5 file, as a byte string

    Returns
    -------
    hid_t
        The file ID
    '''
    # cdef:
    #     int args_num
    #     int* args_num_p
    #     char** args
    #     MPI_Comm comm = MPI_COMM_WORLD
    #     MPI_Info info = MPI_INFO_NULL
    # args_num = 0
    # args_num_p = &args_num

    # NOTE: Turning off error printing here because we already know (and
    #   handle appropriately) exceptions related to existing files
    # H5Eset_auto2(H5E_DEFAULT, <H5E_auto2_t>NULL, NULL)

    # In both cases, using H5P_DEFAULT for args fcpl_id, fapl_id
    if access(filename, F_OK) == 0:
        # If file already exists, delete it; unfortunately, this is necessary
        #   because H5F_ACC_RDWR is determined to be an invalid flag
        remove(filename)

    # MPI_Init(args_num_p, &args)
    # fapl = H5Pcreate(H5P_FILE_ACCESS)
    # H5Pset_fapl_mpio(fapl, comm, info)
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
    cdef hsize_t chunk_size[2]
    cdef float fill_value[1]
    fill_value[0] = <float>-9999
    chunk_size[0] = 300
    chunk_size[1] = 100
    # NOTE: Turning off error printing here because we already know (and
    #   handle appropriately) exceptions related to existing groups
    # H5Eset_auto2(H5E_DEFAULT, <H5E_auto2_t>NULL, NULL)
    # If there is no intermediate group, destination is the file
    dset_name = field.decode('UTF-8')
    dest = fid
    # In case of an intermediate group, i.e., "group/dataset_name"
    if '/' in dset_name:
        group, dset_name = field.decode('UTF-8').split('/')
        if H5Lexists(fid, group.encode('UTF-8'), H5P_DEFAULT) <= 0:
            gid = H5Gcreate(
                fid, group.encode('UTF-8'), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)
        dest = gid

    # Create the data transfer property list;
    #   https://support.hdfgroup.org/documentation/hdf5/latest/_par_compr.html
    # dxpl = H5Pcreate(H5P_DATASET_XFER)
    # H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE)
    # Create the dataset property list
    dcpl = H5Pcreate(H5P_DATASET_CREATE)
    if H5Pset_layout(dcpl, H5D_CHUNKED) < 0:
        print('ERROR in setting layout')
    if H5Pset_chunk(dcpl, 2, chunk_size) < 0:
        print('ERROR in setting chunks')
    if H5Pset_fill_value(dcpl, H5T_IEEE_F32LE, fill_value) < 0:
        print('ERROR in setting the fill vlaue')
    if H5Pset_deflate(dcpl, 5) < 0:
        print('ERROR in setting gzip filter')

    dset_id = H5Dcreate( # Using H5P_DEFAULT for lcpl_id, ..., dapl_id
        dest, dset_name.encode('UTF-8'), dtype, dspace,
        H5P_DEFAULT, dcpl, H5P_DEFAULT)
    if H5Dwrite(dset_id, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buff) < 0:
        print('ERROR in writing HDF5 file')
    H5Dclose(dset_id)
    H5Pclose(dcpl)
    # H5Pclose(dxpl)
