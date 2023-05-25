
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

// copy nElements of SOURCE, beginning at "atSlot", into DEST slot 0, where each
// shares the same datatype (added by laj 06-05-2013)
void *copyUUTA(void *vSource_p, void *vDest_p, const size_t srcSlot,
               const size_t destSlot, const signed int dataType,
               const size_t nElements) {
  TI_ATOM_PTR *src_p = (TI_ATOM_PTR *)
      vSource_p; // ...essential to dereference the void * as a real type here
  TI_ATOM_PTR *dst_p = (TI_ATOM_PTR *)
      vDest_p; // ...essential to dereference the void * as a real type here

  size_t srcOffset = DFKNTsize(dataType) * srcSlot;
  size_t destOffset = DFKNTsize(dataType) * destSlot;
  size_t nBytes = DFKNTsize(dataType) * nElements;

  memcpy(dst_p->aUint8_p + destOffset, src_p->aUint8_p + srcOffset, nBytes);

  // return ptr to destination buffer as a syntactic convenience
  return vDest_p;
} // end::copyUUTA()

// copy nElements of SOURCE, beginning at "atSlot", into DEST slot 0, where each
// shares the same datatype
void *copyUUTA0(void *vDest_p, void *vSource_p, const signed int dataType,
                const size_t atSlot, const size_t nElements) {
  TI_ATOM_PTR *src_p = (TI_ATOM_PTR *)
      vSource_p; // ...essential to dereference the void * as a real type here
  TI_ATOM_PTR *dst_p = (TI_ATOM_PTR *)
      vDest_p; // ...essential to dereference the void * as a real type here

  size_t thisOffset = DFKNTsize(dataType) * atSlot;
  size_t nBytes = DFKNTsize(dataType) * nElements;

  memcpy(dst_p->aUint8_p, src_p->aUint8_p + thisOffset, nBytes);

  // return ptr to destination buffer as a syntactic convenience
  return vDest_p;
} // end::copyUUTA0()

// copyUUTA_to_UUTA is a more flexible, "any-to-any" UUTA operator.
// purpose: Copy nElements of SOURCE, beginning at srcSlot, into DEST, at
// destSlot,
//    where each shares the same datatype. Be CAREFUL using this one, that each
//    UUTA is sized
// to accomodate all the data requested here! Later we'll add an extension that
// internally guards against overcopy via use of a UUTA's nElemAllocated
// property.
void *copyUUTA_to_UUTA(void *vDest_p, void *vSource_p, const signed int dataType,
                       const size_t srcSlot, const size_t destSlot,
                       size_t nElements) {
  TI_ATOM_PTR *src_p = (TI_ATOM_PTR *)
      vSource_p; // ...essential to dereference the void * as a real type here
  TI_ATOM_PTR *dst_p = (TI_ATOM_PTR *)
      vDest_p; // ...essential to dereference the void * as a real type here

  size_t nBytes = DFKNTsize(dataType) * nElements;
  size_t srcOffset = DFKNTsize(dataType) * srcSlot;
  size_t dstOffset = DFKNTsize(dataType) * destSlot;

  memcpy(dst_p->aUint8_p + dstOffset, src_p->aUint8_p + srcOffset, nBytes);

  // return ptr to destination buffer as a syntactic convenience
  return vDest_p;
} // end::copyUUTA_to_UUTA()

// Assign ONE element FROM SOURCE, into DESTINATION UUTA, at position "atSlot",
// where each shares a common dataType Call Example:  float32 myFloat32 =
// -9999.0;  setIntoUUTA(&destUUTA, &myFloat32, DFNT_FLOAT32, arrayIndex);
void *setIntoUUTA(void *vDest_p, void *vSource_p, const signed int dataType,
                  const size_t atSlot) {
  TI_ATOM_PTR *src_p = (TI_ATOM_PTR *)vSource_p;
  TI_ATOM_PTR *dst_p = (TI_ATOM_PTR *)vDest_p;
  size_t thisOffset = DFKNTsize(dataType) * atSlot;

  memcpy((unsigned char *)(dst_p->aUint8_p + thisOffset), (unsigned char *)src_p,
         DFKNTsize(dataType));

  // return ptr to destination buffer as a syntactic convenience
  return vDest_p;
} // end::setIntoUUTA()

// Retrieve ONE element from  UUTA SOURCE, starting at offset, storing in a
// destination (scalar), where each share a common datatype.  A call Example:
//   TI_ATOM_PTR myUUTA ;
//   float32 myFloat32 = -9999.0 ;
//        getFromUUTA(&myUUTA, &myFloat32, DFNT_FLOAT32, arrayIndex);
//
void *getFromUUTA(void *vDest_p, void *vSource_p, const signed int dataType,
                  const size_t atSlot) {
  TI_ATOM_PTR *src_p = (TI_ATOM_PTR *)vSource_p;

  size_t sizOfOne = DFKNTsize(dataType);
  size_t thisOffset =
      sizOfOne * atSlot; // byte offset into source where starting value lies
  // if variable is "myFloat32", call:
  // getFromUUTA(&myFloat32,&sourceUUTA,arrayIndex);
  memcpy(vDest_p, (unsigned char *)src_p->aUint8_p + thisOffset, sizOfOne);
  return (vDest_p);
} // end::getFromUUTA()
