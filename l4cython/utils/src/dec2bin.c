// From SMAP L4_C operational C code; authors: Joe Glassy, Lucas Jones
//
//  Converts a decimal to its binary string, then splits that bit string into
//  sub-units (between start_at and end_at, where start_at is the right-most)
//  bit position. For example:
//
//    >>> bits_from_uint32(3, 7, 80)
//    10
//
//  The binary representation of 80 is 01010000 and the sub-unit from 3 to 7
//  is 01010, which has a decimal representation of 10.
unsigned long int bits_from_uint32(
  const signed long int start_at, const signed long int end_at, unsigned long int value)
{
 signed long int p = end_at,              // anchor the start here
                 n = (end_at-start_at)+1; // sum of bits (width) to retrieve

 return (value >> (p+1-n)) & ~(~0 << n);

} // bits_from_uint32
