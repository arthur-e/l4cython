# cython: language_level=3
# distutils: include_dirs = ["src/"]

cdef void read_hdf5(char* filename, char* field, void* buff):
    # Convert the unicode filename to a C string
    filename_byte_string = filename.encode('UTF-8')
    field_byte_string = field.encode('UTF-8')

    # NOTE: In order of arguments...
    #   0 should be equivalent to hex code 0x0000u ("H5F_ACC_RDONLY")
    #   0 is the definition of H5P_DEFAULT in H5Ppublic.h
    fid = H5Fopen(filename_byte_string, 0, H5P_DEFAULT)
    dsetid = H5Dopen1(fid, field_byte_string)
    ret = H5Dread(dsetid, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buff)
