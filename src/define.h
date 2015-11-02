// define.h write by scott.zgeng 2015.10.13


#ifndef  __DEFINE_H__
#define  __DEFINE_H__



typedef bool mv_bool;
typedef char mv_char;
typedef unsigned char mv_byte;
typedef int mv_result;

typedef char mv_int8;
typedef unsigned char mv_uint8;
typedef short mv_int16;
typedef unsigned short mv_uint16;
typedef int mv_int32;
typedef unsigned int mv_uint32;
typedef long long mv_int64;
typedef unsigned long long mv_uint64;
typedef float mv_float;
typedef double mv_double;


#define MV_FAILED       (-1)
#define MV_SUCCEEDED    (0)


// fundamental constants 
#define MV_PI           (3.1415926535897932384626433832795)
#define MV_PI2          (MV_PI*2.0)
#define MV_LOG2         (0.69314718055994530941723212145818)

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif



#endif //__DEFINE_H__

