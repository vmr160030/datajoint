#include <cstdint>
//#include <math.h>
// Common information for random number generation
// *** Note: switching RNGs means that performing mod operations// on the RN could be wrong.  For example, mod on the mac RNG
// is a problem because it has an odd number of possible states.
// even numbers for java make it easier to mod by a lut size such as 2.

#define RANDMACSHORT_MIN (-32767)
#define RANDMACSHORT_MAX (+32767)
#define RANDMACSHORT_SCALE (65535.0)
#define RANDJAVA_MIN (-2147483648)
#define RANDJAVA_MAX (2147483647)
#define RANDJAVA_SCALE (4294967296.0)
#define RANDJAVASHORT_MIN (-32768)
#define RANDJAVASHORT_MAX (32767)
#define RANDMERSENNE_MAX (4294967295)
#define RANDMERSENNE_MIN (0)
#define RANDMERSENNE_SCALE (4294967296.0)

// prior to 2004-10-20 EJC
//#define MIN_RANDOM (-32767)
//#define MAX_RANDOM (+32767)
//#define RANDOM_RANGE (65536)

typedef uint32_t RandMersenneState[625];
typedef int64_t RandJavaState[1];
typedef int32_t RandMacState[1];

#define RAND_MAC 0
#define RAND_JAVA 1
#define RAND_MERSENNE 2
#define ROUND(x) (floor((x)+0.5))
