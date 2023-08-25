
#include "spland.h"

/*DEFLATE function for 9km grid*/
void spland_deflate_9km(spland_ref_struct SPLAND, void *src_p, void *dest_p,
                        const unsigned int dataType) {

  unsigned int l;

  size_t srcSlot;
  size_t destSlot;

  size_t nelem = 1;

  /*Transfer cells from 2D 9km grid to 9km sparse land vector*/
  for (l = 0; l < LLAND9KM; l++) {

    srcSlot = (size_t)M_2D_B0(SPLAND.row[l], SPLAND.col[l], NCOL9KM);
    destSlot = (size_t)l;

    copyUUTA(src_p, dest_p, srcSlot, destSlot, dataType, nelem);

  } // end:: land vec loop
}

/*DEFLATE function for nested 1km grid*/
void spland_deflate_1km(spland_ref_struct SPLAND, void *src_p, void *dest_p,
                        const unsigned int dataType) {

  unsigned int l;

  unsigned char rn, cn;
  unsigned char serialIdx_nested;

  size_t srcSlot;
  size_t destSlot;

  size_t nelem = 1;

  /*Transfer cells from 2D 9km grid to 9km sparse land vector*/
  for (l = 0; l < LLAND9KM; l++) {
    for (rn = 0; rn < NESTED_1KM_IN_9KM; rn++) {
      for (cn = 0; cn < NESTED_1KM_IN_9KM; cn++) {

        // reference 1km nested grid
        srcSlot = (size_t)M_2D_N9_B0(SPLAND.row[l], SPLAND.col[l], rn, cn,
                                     NESTED_1KM_IN_9KM, NCOL1KM);

        // keep 1km nested cells in same relative orientation
        serialIdx_nested = M_2D_B0(rn, cn, NESTED_1KM_IN_9KM);
        destSlot = (size_t)M_2D_B0(l, serialIdx_nested, NESTED_1KM_IN_9KM_SQ);

        // transfer data element from global grid to land vector
        copyUUTA(src_p, dest_p, srcSlot, destSlot, dataType, nelem);

      } // end:: col loop nested 1km
    }   // end:: row loop nested 1km
  }     // end:: land vec loop
}

/*INFLATE function for 9km grid*/
void spland_inflate_9km(spland_ref_struct SPLAND, void *src_p, void *dest_p,
                        const unsigned int dataType) {

  unsigned int l;

  size_t srcSlot;
  size_t destSlot;

  size_t nelem = 1;

  /*Transfer cells from 2D 9km grid to 9km sparse land vector*/
  for (l = 0; l < LLAND9KM; l++) {

    destSlot = (size_t)M_2D_B0(SPLAND.row[l], SPLAND.col[l], NCOL9KM);
    srcSlot = (size_t)l;

    copyUUTA(src_p, dest_p, srcSlot, destSlot, dataType, nelem);

  } // end:: land vec loop
}

/*INFLATE function for nested 1km grid*/
/*NOTE: Really just the same as the deflate functions with the source and dest
 * (and their respective indicies) swapped*/
void spland_inflate_1km(spland_ref_struct SPLAND, void *src_p, void *dest_p,
                        const unsigned int dataType) {

  unsigned int l;

  unsigned char rn, cn;
  unsigned char serialIdx_nested;

  size_t srcSlot;
  size_t destSlot;

  size_t nelem = 1;

  /*Transfer cells from 9km sparse land vector to 2D 9km grid*/
  for (l = 0; l < LLAND9KM; l++) {
    for (rn = 0; rn < NESTED_1KM_IN_9KM; rn++) {
      for (cn = 0; cn < NESTED_1KM_IN_9KM; cn++) {

        // reference 1km nested grid
        destSlot = (size_t)M_2D_N9_B0(SPLAND.row[l], SPLAND.col[l], rn, cn,
                                      NESTED_1KM_IN_9KM, NCOL1KM);

        // keep 1km nested cells in same relative orientation
        serialIdx_nested = M_2D_B0(rn, cn, NESTED_1KM_IN_9KM);
        srcSlot = (size_t)M_2D_B0(l, serialIdx_nested, NESTED_1KM_IN_9KM_SQ);

        // transfer data element from global grid to land vector
        copyUUTA(src_p, dest_p, srcSlot, destSlot, dataType, nelem);

      } // end:: col loop nested 1km
    }   // end:: row loop nested 1km
  }     // end:: land vec 9km loop
}

/*Function for loading ancil row/col data for sparse format*/
int spland_load_9km_rc(spland_ref_struct *SPLAND) {

  FILE *fid;

  fid = fopen(LAND_R_FILE, "rb");

  if (fid != NULL) {

    fread(SPLAND->row, sizeof(unsigned short), LLAND9KM, fid);
    fclose(fid);
  }

  fid = fopen(LAND_C_FILE, "rb");

  if (fid != NULL) {

    fread(SPLAND->col, sizeof(unsigned short), LLAND9KM, fid);
    fclose(fid);
  }

  return (0);
}

/*Initialize the inflation matrix with fill values*/
void spland_inflate_init_9km(void *dest_p, const uint32 dataType) {

  uint32 r, c;

  size_t destSlot;

  for (r = 0; r < NROW9KM; r++) {
    for (c = 0; c < NCOL9KM; c++) {

      destSlot = (size_t)M_2D_B0(r, c, NCOL9KM);

      set_fillval_UUTA(dest_p, dataType, destSlot);

    } // end:: col 9km loop
  }   // end:: row 9km loop
}

void spland_inflate_init_1km(void *dest_p, const uint32 dataType) {

  uint32 r, c, rn, cn;

  size_t destSlot;

  for (r = 0; r < NROW9KM; r++) {
    for (c = 0; c < NCOL9KM; c++) {
      for (rn = 0; rn < NESTED_1KM_IN_9KM; rn++) {
        for (cn = 0; cn < NESTED_1KM_IN_9KM; cn++) {

          // reference 1km nested grid
          destSlot =
              (size_t)M_2D_N9_B0(r, c, rn, cn, NESTED_1KM_IN_9KM, NCOL1KM);

          set_fillval_UUTA(dest_p, dataType, destSlot);

        } // end:: col 1km nested loop
      }   // end:: row 1km nested loop

    } // end:: col 9km loop
  }   // end:: row 9km loop
}

/*Set fill value for an array of a given data type at the specified location in
 * the array*/
void set_fillval_UUTA(void *vDest_p, const int32 dataType,
                      const size_t atSlot) {

  TI_ATOM_PTR *dst_p = (TI_ATOM_PTR *)vDest_p;

  size_t i = atSlot;

  // set fill value based on data type
  switch (dataType) {
  case isChar:
    dst_p->aInt8_p[i] = (signed char)-127;
    break;
  case isUint8:
    dst_p->aUint8_p[i] = (unsigned char)255;
    break;
  case isInt16:
    dst_p->aInt16_p[i] = (signed short)-32767;
    break;
  case isUint16:
    dst_p->aUint16_p[i] = (unsigned short)65535;
    break;
  case isInt32:
    dst_p->aInt32_p[i] = (signed int)-2.1474e9;
    break;
  case isUint32:
    dst_p->aUint32_p[i] = (unsigned int)-4.2950e9;
    break;
  case isInt64:
    dst_p->aInt64_p[i] = (signed long)-9.2233e18;
    break;
  case isUint64:
    dst_p->aUint64_p[i] = (unsigned long)1.8446e19;
    break;
  case isFloat32:
    dst_p->aFlt32_p[i] = (float)-9.999e9;
    break;
  case isFloat64:
    dst_p->aFlt64_p[i] = (double)-9.999e9;
    break;
  } // end switch on datatype
}


/************************************************************
 * Substitute for HDF4's DFKNTsize()
 *   Determine the size, given the number type
 ************************************************************/
int size_in_bytes(int32 number_type) {
    // No support for little-endian machines
    switch (number_type) {
        /* HDF types */
        case DFNT_UCHAR:
            return (SIZE_UCHAR);
        case DFNT_CHAR:
            return (SIZE_CHAR);
        case DFNT_INT8:
            return (SIZE_INT8);
        case DFNT_UINT8:
            return (SIZE_UINT8);
        case DFNT_INT16:
            return (SIZE_INT16);
        case DFNT_UINT16:
            return (SIZE_UINT16);
        case DFNT_INT32:
            return (SIZE_INT32);
        case DFNT_UINT32:
            return (SIZE_UINT32);
        case DFNT_FLOAT32:
            return (SIZE_FLOAT32);
        case DFNT_FLOAT64:
            return (SIZE_FLOAT64);
        /* Unknown types */
        default:
            break;
    } /* switch */
    return -1;
}
