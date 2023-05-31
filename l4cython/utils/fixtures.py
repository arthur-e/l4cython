# From hntdefs.h
DFNT_FLOAT32 = 5
DFNT_FLOAT64 = 6
DFNT_INT8   = 20
DFNT_UINT8  = 21
DFNT_INT16  = 22
DFNT_UINT16 = 23
DFNT_INT32  = 24
DFNT_UINT32 = 25
DFNT_INT64  = 26
DFNT_UINT64 = 27

# EASE-Grid 2.0 properties
NCOL9KM = 3856
NROW9KM = 1624
NCOL1KM = 34704
NROW1KM = 14616
M01_NESTED_IN_M09 = 9 * 9
SPARSE_M09_N = 1664040 # Number of grid cells in sparse ("land") arrays
SPARSE_M01_N = M01_NESTED_IN_M09 * SPARSE_M09_N

READ  = 'rb'.encode('UTF-8') # Binary read mode as byte string
WRITE = 'wb'.encode('UTF-8') # Binary write mode as byte string
