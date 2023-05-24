
#include "uuta.h"

// copy nElements of SOURCE, beginning at "atSlot", into DEST slot 0, where each
// shares the same datatype (added by laj 06-05-2013)
void *copyUUTA(void *vSource_p, void *vDest_p, const size_t srcSlot,
               const size_t destSlot, const int32 dataType,
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
void *copyUUTA0(void *vDest_p, void *vSource_p, const int32 dataType,
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
void *copyUUTA_to_UUTA(void *vDest_p, void *vSource_p, const int32 dataType,
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
void *setIntoUUTA(void *vDest_p, void *vSource_p, const int32 dataType,
                  const size_t atSlot) {
  TI_ATOM_PTR *src_p = (TI_ATOM_PTR *)vSource_p;
  TI_ATOM_PTR *dst_p = (TI_ATOM_PTR *)vDest_p;
  size_t thisOffset = DFKNTsize(dataType) * atSlot;

  memcpy((uint8 *)(dst_p->aUint8_p + thisOffset), (uint8 *)src_p,
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
void *getFromUUTA(void *vDest_p, void *vSource_p, const int32 dataType,
                  const size_t atSlot) {
  TI_ATOM_PTR *src_p = (TI_ATOM_PTR *)vSource_p;

  size_t sizOfOne = DFKNTsize(dataType);
  size_t thisOffset =
      sizOfOne * atSlot; // byte offset into source where starting value lies
  // if variable is "myFloat32", call:
  // getFromUUTA(&myFloat32,&sourceUUTA,arrayIndex);
  memcpy(vDest_p, (uint8 *)src_p->aUint8_p + thisOffset, sizOfOne);
  return (vDest_p);
} // end::getFromUUTA()
