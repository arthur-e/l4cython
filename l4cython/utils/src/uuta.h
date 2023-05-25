

/*
 test_uuta.c
 purpose  : evaluate correctness of "Universal Union of Typeful Atoms" UUTA
 transforms revision : v1.0-2013-07-03T15:04:00,barley,jmg author   : joe glassy

 build    : gcc -g -Wall -std=c99 -o typinde.ex typinde.c -I/usr/include/hdf
 -L/usr/lib64/hdf -lm -lmfhdf -ldf -lefence : Using the Electric Fence library
 (e.g. -lefence) is optional

 NOTES:
 --there are really TWO types of "offsets" commonly used, and its critical to
 distinguish these:

   a) a "byte-wise" offset, useful for handling transformations of ANY
 datatype,e.g. Here, "nBytes" is a byte-wise offset: size_t nBytes =
 DFKNTsize(DFNT_FLOAT32) * nElements ; memcpy(destBuffer_p, sourceBuffer_p,
 nBytes);

   b) an "element-wise" offset (e.g. "index"), useful for a transformation
 involving a single datatype, e.g. an array "index" is an example of an
 "element-wise" index: myArray[index] = 234.3 ;

 TEST OUTPUTS:
  Show the unique (different) starting memory addresses of the 2 UUTA's
  Buffer addresses (start): Src 0x2b2710659fe0 Dest 0x2b271065bfe0

*/

#ifndef HEADER_UUTA_DEF
#define HEADER_UUTA_DEF 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// where WORDSIZE is correctly established
#include <sys/types.h>
// where int64_t, uint64_t are now defined
#include <stdint.h>

#include <assert.h>
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <time.h>

#include <stddef.h>
#include <unistd.h>

// other useful resources
// for fstat() or stat() need sys/stat.h
#include <sys/stat.h>
// #include <process.h>  ..for process id (pid,ppid,gpid etc)
#include <stddef.h>
// stdint.h is where int64_t and uint64_t are now defined
#include <stdint.h>
// where __WORDSIZE==64, in POSIX 2.6, this is where scalar types like int64_t
// are echoed
#include <sys/types.h>

#include "hdf.h"
#include "mfhdf.h"

#define VersTag "v1.0-2013-07-03T15:04:00,barley,jmg"

// typedef int64_t signed long;
// typedef uint64_t unsigned long;

// type-independent union (translation-layer) of pointers to atomic types
typedef union {
  signed char *aInt8_p;
  unsigned char *aUint8_p;
  signed short *aInt16_p;
  unsigned short *aUint16_p;
  signed int *aInt32_p;
  unsigned int *aUint32_p;
  signed long *aInt64_p;
  unsigned long *aUint64_p;
  float *aFlt32_p;
  double *aFlt64_p;
  void *aVoid_p;
} TI_ATOM_PTR;

//  --by "atSlot" we refer to an array index, and NOT to a byte offset.
//  --use: sizOfOne = DFKNTsize(DFNT_FLOAT32) to determine n. bytes per single
//  scalar object
//         of a given supported datatype

// copy nElements of SOURCE, beginning at "atSlot", into DEST slot 0, where each
// shares the same datatype (added by laj 06-05-2013)
void *copyUUTA(void *vSource_p, void *vDest_p, const size_t srcSlot,
               const size_t destSlot, const signed int dataType,
               const size_t nElements);

// NEW UUTA FUNCTIONS    2013-07/03,jmg
// copy data from a SOURCE array of datatype, to a destination array of same
// datatype, and rtn ptr to new dest as well
void *copyUUTA0(void *vDest_p, void *vSource_p, const signed int dataType,
                const size_t atSlot, const size_t nElements);

// copy nElements of SOURCE UUTA, starting at SrcOffset, into DESTINATION UUTA,
// at DestOffset.
void *copyUUTA_to_UUTA(void *vDest_p, void *vSource_p, const signed int dataType,
                       const size_t srcSlot, const size_t dstSlot,
                       size_t nElements);

// Assign ONE element of SOURCE, into DESTINATION UUTA at position "atSlot",
// where each shares a common dataType
void *setIntoUUTA(void *vDest_p, void *vSource_p, const signed int dataType,
                  const size_t atSlot);

// Retrieve ONE element from  UUTA SOURCE, starting at offset, storing in a
// destination (scalar), where each share a common datatype.  A call Example:
void *getFromUUTA(void *vDest_p, void *vSource_p, const signed int dataType,
                  const size_t atSlot);

#endif
