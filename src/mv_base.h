// mv_base.h write by scott.zgeng 2015.9.19
// 用于移植 opensift使用了opencv的部分


#ifndef  __MV_BASE_H__
#define  __MV_BASE_H__



#include <stdlib.h>



#include "define.h"





#define MV_CN_MAX     512
#define MV_CN_SHIFT   3
#define MV_DEPTH_MAX  (1 << MV_CN_SHIFT)

#define MV_8U   0
#define MV_8S   1
#define MV_16U  2
#define MV_16S  3
#define MV_32S  4
#define MV_32F  5
#define MV_64F  6
#define MV_USRTYPE1 7

#define MV_MAT_DEPTH_MASK       (MV_DEPTH_MAX - 1)
#define MV_MAT_DEPTH(flags)     ((flags) & MV_MAT_DEPTH_MASK)

#define MV_MAKETYPE(depth,cn) (MV_MAT_DEPTH(depth) + (((cn)-1) << MV_CN_SHIFT))


#define MV_8UC1 MV_MAKETYPE(MV_8U,1)
#define MV_8UC2 MV_MAKETYPE(MV_8U,2)
#define MV_8UC3 MV_MAKETYPE(MV_8U,3)
#define MV_8UC4 MV_MAKETYPE(MV_8U,4)
#define MV_8UC(n) MV_MAKETYPE(MV_8U,(n))

#define MV_8SC1 MV_MAKETYPE(MV_8S,1)
#define MV_8SC2 MV_MAKETYPE(MV_8S,2)
#define MV_8SC3 MV_MAKETYPE(MV_8S,3)
#define MV_8SC4 MV_MAKETYPE(MV_8S,4)
#define MV_8SC(n) MV_MAKETYPE(MV_8S,(n))

#define MV_16UC1 MV_MAKETYPE(MV_16U,1)
#define MV_16UC2 MV_MAKETYPE(MV_16U,2)
#define MV_16UC3 MV_MAKETYPE(MV_16U,3)
#define MV_16UC4 MV_MAKETYPE(MV_16U,4)
#define MV_16UC(n) MV_MAKETYPE(MV_16U,(n))

#define MV_16SC1 MV_MAKETYPE(MV_16S,1)
#define MV_16SC2 MV_MAKETYPE(MV_16S,2)
#define MV_16SC3 MV_MAKETYPE(MV_16S,3)
#define MV_16SC4 MV_MAKETYPE(MV_16S,4)
#define MV_16SC(n) MV_MAKETYPE(MV_16S,(n))

#define MV_32SC1 MV_MAKETYPE(MV_32S,1)
#define MV_32SC2 MV_MAKETYPE(MV_32S,2)
#define MV_32SC3 MV_MAKETYPE(MV_32S,3)
#define MV_32SC4 MV_MAKETYPE(MV_32S,4)
#define MV_32SC(n) MV_MAKETYPE(MV_32S,(n))

#define MV_32FC1 MV_MAKETYPE(MV_32F,1)
#define MV_32FC2 MV_MAKETYPE(MV_32F,2)
#define MV_32FC3 MV_MAKETYPE(MV_32F,3)
#define MV_32FC4 MV_MAKETYPE(MV_32F,4)
#define MV_32FC(n) MV_MAKETYPE(MV_32F,(n))

#define MV_64FC1 MV_MAKETYPE(MV_64F,1)
#define MV_64FC2 MV_MAKETYPE(MV_64F,2)
#define MV_64FC3 MV_MAKETYPE(MV_64F,3)
#define MV_64FC4 MV_MAKETYPE(MV_64F,4)
#define MV_64FC(n) MV_MAKETYPE(MV_64F,(n))

#define MV_MAT_CN_MASK          ((MV_CN_MAX - 1) << MV_CN_SHIFT)
#define MV_MAT_CN(flags)        ((((flags) & MV_MAT_CN_MASK) >> MV_CN_SHIFT) + 1)
#define MV_MAT_TYPE_MASK        (MV_DEPTH_MAX*MV_CN_MAX - 1)
#define MV_MAT_TYPE(flags)      ((flags) & MV_MAT_TYPE_MASK)
#define MV_MAT_CONT_FLAG_SHIFT  14
#define MV_MAT_CONT_FLAG        (1 << MV_MAT_CONT_FLAG_SHIFT)
#define MV_IS_MAT_CONT(flags)   ((flags) & MV_MAT_CONT_FLAG)
#define MV_IS_CONT_MAT          MV_IS_MAT_CONT
#define MV_SUBMAT_FLAG_SHIFT    15
#define MV_SUBMAT_FLAG          (1 << MV_SUBMAT_FLAG_SHIFT)


#define MV_MAGIC_MASK       0xFFFF0000
#define MV_MAT_MAGIC_VAL    0x42420000


/* 0x3a50 = 11 10 10 01 01 00 00 ~ array of log2(sizeof(arr_type_elem)) */
#define MV_ELEM_SIZE(type) \
    (MV_MAT_CN(type) << ((((sizeof(size_t) / 4 + 1) * 16384 | 0x3a50) >> MV_MAT_DEPTH(type) * 2) & 3))



#define IPL_DEPTH_SIGN 0x80000000

#define IPL_DEPTH_1U     1
#define IPL_DEPTH_8U     8
#define IPL_DEPTH_16U   16
#define IPL_DEPTH_32F   32

#define IPL_DEPTH_8S  (IPL_DEPTH_SIGN| 8)
#define IPL_DEPTH_16S (IPL_DEPTH_SIGN|16)
#define IPL_DEPTH_32S (IPL_DEPTH_SIGN|32)

#define IPL_DATA_ORDER_PIXEL  0
#define IPL_DATA_ORDER_PLANE  1

#define IPL_ORIGIN_TL 0
#define IPL_ORIGIN_BL 1

#define IPL_ALIGN_4BYTES   4
#define IPL_ALIGN_8BYTES   8
#define IPL_ALIGN_16BYTES 16
#define IPL_ALIGN_32BYTES 32

#define IPL_ALIGN_DWORD   IPL_ALIGN_4BYTES
#define IPL_ALIGN_QWORD   IPL_ALIGN_8BYTES

#define IPL_BORDER_CONSTANT   0
#define IPL_BORDER_REPLICATE  1
#define IPL_BORDER_REFLECT    2
#define IPL_BORDER_WRAP       3



//#define MV_INTER_NN 0
#define MV_INTER_LINEAR  1
//#define MV_INTER_CUBIC  2
//#define MV_INTER_AREA  3
//#define MV_INTER_LANCZOS4  4

#define MV_WARP_FILL_OUTLIERS  8
#define MV_WARP_INVERSE_MAP 16







/** linear convolution with \f$\texttt{size1}\times\texttt{size2}\f$ box kernel (all 1's). If
you want to smooth different pixels with different-size box kernels, you can use the integral
image that is computed using integral */
#define MV_BLUR_NO_SCALE 0
#define  MV_BLUR 1
#define  MV_GAUSSIAN 2
#define  MV_MEDIAN 3
#define  MV_BILATERAL 4


#define MV_WHOLE_SEQ_END_INDEX 0x3fffffff
#define MV_WHOLE_SEQ  mv_slice_t(0, MV_WHOLE_SEQ_END_INDEX)


#define MV_LU  0
#define MV_SVD 1
#define MV_SVD_SYM 2
#define MV_CHOLESKY 3
#define MV_QR  4
#define MV_NORMAL 16


#define MV_SVD_MODIFY_A   1
#define MV_SVD_U_T        2
#define MV_SVD_V_T        4


#define MV_GEMM_A_T 1
#define MV_GEMM_B_T 2
#define MV_GEMM_C_T 4


#define MV_AUTOSTEP  0x7fffffff




struct mv_point_t
{
    int x;
    int y;

    mv_point_t(int _x, int _y) {
        x = _x;
        y = _y;
    }

    mv_point_t() {
        x = 0;
        y = 0;
    }
};


struct mv_point_d_t
{
    double x;
    double y;

    mv_point_d_t() {
        x = 0;
        y = 0;
    }
    mv_point_d_t(double _x, double _y) {
        x = _x;
        y = _y;
    }
};


struct mv_rect_t
{
    int x;
    int y;
    int width;
    int height;

    mv_rect_t() {
        x = 0;
        y = 0;
        width = 0;
        height = 0;
    }

    mv_rect_t(int x_, int y_, int width_, int height_) {
        x = x_;
        y = y_;
        width = width_;
        height = height_;
    }
};


struct mv_scalar_t
{
    double val[4];
    mv_scalar_t(double d1, double d2, double d3, double d4) {
        val[0] = d1; 
        val[1] = d2;
        val[2] = d3;
        val[3] = d4;
    }

    mv_scalar_t(double d) {
        val[0] = d;
        val[1] = d;
        val[2] = d;
        val[3] = d;
    }
};

#define MV_RGB(r, g, b )  mv_scalar_t((b), (g), (r), 0)


struct mv_size_t
{
    int width;
    int height;

    mv_size_t() {
        width = 0;
        height = 0;
    }

    mv_size_t(int w, int h) {
        width = w;
        height = h;
    }
};




//struct mv_matrix_t
//{
//    int type;
//    int step;
//
//    // for internal use only 
//    int* refcount;
//    int hdr_refcount;
//
//    union
//    {
//        unsigned char* ptr;
//        short* s;
//        int* i;
//        float* fl;
//        double* db;
//    } data;
//
//    int rows;
//    int cols;
//
//
//    mv_matrix_t() {
//        type = 0;
//        step = 0;
//
//        cols = 0;
//        rows = 0;
//        
//        data.ptr = NULL;
//
//        refcount = NULL;
//        hdr_refcount = 0;
//    }
//
//    mv_matrix_t(int r, int c, int t, void* ptr)
//    {        
//        type = MV_MAT_TYPE(t);
//        type = MV_MAT_MAGIC_VAL | MV_MAT_CONT_FLAG | t;
//        cols = c;
//        rows = r;
//        step = c * MV_ELEM_SIZE(t);
//        data.ptr = (unsigned char*)ptr;
//        refcount = NULL;
//        hdr_refcount = 0; 
//    }
//};



//
//class resize_cubic
//{
//    static const int MAX_ESIZE = 16;
//    static const int INTER_RESIZE_COEF_BITS = 11;
//    static const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;
//
//
//public:
//    resize_cubic() {}
//    ~resize_cubic() {}
//
//public:
//    mv_result init(mv_size_t ssize, mv_size_t dsize);
//    void operator()(const mv_image_t* src, mv_image_t* dst);
//
//
//private:
//    inline void interpolate_cubic(float x, float* coeffs)
//    {
//        const float A = -0.75f;
//
//        coeffs[0] = ((A*(x + 1) - 5 * A)*(x + 1) + 8 * A)*(x + 1) - 4 * A;
//        coeffs[1] = ((A + 2)*x - (A + 3))*x*x + 1;
//        coeffs[2] = ((A + 2)*(1 - x) - (A + 3))*(1 - x)*(1 - x) + 1;
//        coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
//    }
//
//
//private:
//    int* m_xofs;
//    int* m_yofs;
//    short* m_alpha;
//    short* m_beta;
//    int m_xmin;
//    int m_xmax;
//    int m_ksize;
//    mv_size_t m_ssize;
//    mv_size_t m_dsize;
//
//    double m_scale_x;
//    double m_scale_y;
//
//    float m_cbuf[MAX_ESIZE];
//};
//
//


/** @brief Rounds floating-point number to the nearest integer

@param value floating-point number. If the value is outside of INT_MIN ... INT_MAX range, the
result is not defined.
*/
inline int mv_round(double value)
{
#if ((defined _MSC_VER && defined _M_X64) || (defined __GNUC__ && defined __x86_64__ \
    && defined __SSE2__ && !defined __APPLE__)) && !defined(__CUDACC__)
    __m128d t = _mm_set_sd(value);
    return _mm_cvtsd_si32(t);
#elif defined _MSC_VER && defined _M_IX86
    int t;
    __asm
    {
        fld value;
        fistp t;
    }
    return t;
#elif ((defined _MSC_VER && defined _M_ARM) || defined MV_ICC || \
    defined __GNUC__) && defined HAVE_TEGRA_OPTIMIZATION
    TEGRA_ROUND_DBL(value);
#elif defined MV_ICC || defined __GNUC__
# if MV_VFP
    ARM_ROUND_DBL(value);
# else
    return (int)lrint(value);
# endif
#else
    /* it's ok if round does not comply with IEEE754 standard;
    the tests should allow +/-1 difference when the tested functions use round */
    return (int)(value + (value >= 0 ? 0.5 : -0.5));
#endif
}


inline int mv_ceil(double value)
{
#if (defined _MSC_VER && defined _M_X64 || (defined __GNUC__ && defined __SSE2__&& !defined __APPLE__)) && !defined(__CUDACC__)
    __m128d t = _mm_set_sd(value);
    int i = _mm_cvtsd_si32(t);
    return i + _mm_movemask_pd(_mm_cmplt_sd(_mm_cvtsi32_sd(t, i), t));
#elif defined __GNUC__
    int i = (int)value;
    return i + (i < value);
#else
    int i = mv_round(value);
    float diff = (float)(i - value);
    return i + (diff < 0);
#endif
}


/** @brief Rounds floating-point number to the nearest integer not larger than the original.

The function computes an integer i such that:
\f[i \le \texttt{value} < i+1\f]
@param value floating-point number. If the value is outside of INT_MIN ... INT_MAX range, the
result is not defined.
*/
inline int mv_floor(double value)
{
#if (defined _MSC_VER && defined _M_X64 || (defined __GNUC__ && defined __SSE2__ && !defined __APPLE__)) && !defined(__CUDACC__)
    __m128d t = _mm_set_sd(value);
    int i = _mm_cvtsd_si32(t);
    return i - _mm_movemask_pd(_mm_cmplt_sd(t, _mm_cvtsi32_sd(t, i)));
#elif defined __GNUC__
    int i = (int)value;
    return i - (i > value);
#else
    int i = mv_round(value);
    float diff = (float)(value - i);
    return i - (diff < 0);
#endif
}








/////////////// saturate_cast (used in image & signal processing) ///////////////////

/**
Template function for accurate conversion from one primitive type to another.

The functions saturate_cast resemble the standard C++ cast operations, such as static_cast\<T\>()
and others. They perform an efficient and accurate conversion from one primitive type to another
(see the introduction chapter). saturate in the name means that when the input value v is out of the
range of the target type, the result is not formed just by taking low bits of the input, but instead
the value is clipped. For example:
@code
uchar a = saturate_cast<uchar>(-100); // a = 0 (UCHAR_MIN)
short b = saturate_cast<short>(33333.33333); // b = 32767 (SHRT_MAX)
@endcode
Such clipping is done when the target type is unsigned char , signed char , unsigned short or
signed short . For 32-bit integers, no clipping is done.

When the parameter is a floating-point value and the target type is an integer (8-, 16- or 32-bit),
the floating-point value is first rounded to the nearest integer and then clipped if needed (when
the target type is 8- or 16-bit).

This operation is used in the simplest or most complex image processing functions in OpenCV.
*/

template<typename _Tp> static inline _Tp saturate_cast(mv_int8 v)   { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(mv_uint8 v)  { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(mv_int16 v)  { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(mv_uint16 v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(mv_int32 v)  { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(mv_uint32 v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(mv_int64 v)  { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(mv_uint64 v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(mv_float v)  { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(mv_double v) { return _Tp(v); }



template<> inline mv_int8 saturate_cast<mv_int8>(mv_uint8 v)        { return (mv_int8)MIN((mv_int32)v, SCHAR_MAX); }
template<> inline mv_int8 saturate_cast<mv_int8>(mv_int16 v)        { return (mv_int8)((mv_uint32)(v - SCHAR_MIN) <= (mv_uint32)UCHAR_MAX ? v : v > 0 ? SCHAR_MAX : SCHAR_MIN); }
template<> inline mv_int8 saturate_cast<mv_int8>(mv_uint16 v)       { return (mv_int8)MIN((mv_uint32)v, (mv_uint32)SCHAR_MAX); }
template<> inline mv_int8 saturate_cast<mv_int8>(mv_int32 v)        { return (mv_int8)((mv_uint32)(v - SCHAR_MIN) <= (mv_uint32)UCHAR_MAX ? v : v > 0 ? SCHAR_MAX : SCHAR_MIN); }
template<> inline mv_int8 saturate_cast<mv_int8>(mv_uint32 v)       { return (mv_int8)MIN(v, (mv_uint32)SCHAR_MAX); }
template<> inline mv_int8 saturate_cast<mv_int8>(mv_int64 v)        { return (mv_int8)((mv_uint64)((mv_int64)v - SCHAR_MIN) <= (mv_uint64)UCHAR_MAX ? v : v > 0 ? SCHAR_MAX : SCHAR_MIN); }
template<> inline mv_int8 saturate_cast<mv_int8>(mv_uint64 v)       { return (mv_int8)MIN(v, (mv_uint64)SCHAR_MAX); }
template<> inline mv_int8 saturate_cast<mv_int8>(mv_float v)        { mv_int32 iv = mv_round(v); return saturate_cast<mv_int8>(iv); }
template<> inline mv_int8 saturate_cast<mv_int8>(mv_double v)       { mv_int32 iv = mv_round(v); return saturate_cast<mv_int8>(iv); }


template<> inline mv_uint8 saturate_cast<mv_uint8>(mv_int8 v)       { return (mv_uint8)MAX((mv_int32)v, 0); }
template<> inline mv_uint8 saturate_cast<mv_uint8>(mv_int16 v)      { return (mv_uint8)((mv_uint32)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0); }
template<> inline mv_uint8 saturate_cast<mv_uint8>(mv_uint16 v)     { return (mv_uint8)MIN((mv_uint32)v, (mv_uint32)UCHAR_MAX); }
template<> inline mv_uint8 saturate_cast<mv_uint8>(mv_int32 v)      { return (mv_uint8)((mv_uint32)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0); }
template<> inline mv_uint8 saturate_cast<mv_uint8>(mv_uint32 v)     { return (mv_uint8)MIN(v, (mv_uint32)UCHAR_MAX); }
template<> inline mv_uint8 saturate_cast<mv_uint8>(mv_int64 v)      { return (mv_uint8)((mv_uint64)v <= (mv_uint64)UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0); }
template<> inline mv_uint8 saturate_cast<mv_uint8>(mv_uint64 v)     { return (mv_uint8)MIN(v, (mv_uint64)UCHAR_MAX); }
template<> inline mv_uint8 saturate_cast<mv_uint8>(mv_float v)      { mv_int32 iv = mv_round(v); return saturate_cast<mv_uint8>(iv); }
template<> inline mv_uint8 saturate_cast<mv_uint8>(mv_double v)     { mv_int32 iv = mv_round(v); return saturate_cast<mv_uint8>(iv); }


template<> inline mv_int16 saturate_cast<mv_int16>(mv_uint16 v)     { return (mv_int16)MIN((mv_int32)v, SHRT_MAX); }
template<> inline mv_int16 saturate_cast<mv_int16>(mv_int32 v)      { return (mv_int16)((mv_uint32)(v - SHRT_MIN) <= (mv_uint32)USHRT_MAX ? v : v > 0 ? SHRT_MAX : SHRT_MIN); }
template<> inline mv_int16 saturate_cast<mv_int16>(mv_uint32 v)     { return (mv_int16)MIN(v, (mv_uint32)SHRT_MAX); }
template<> inline mv_int16 saturate_cast<mv_int16>(mv_int64 v)      { return (mv_int16)((mv_uint64)((mv_int64)v - SHRT_MIN) <= (mv_uint64)USHRT_MAX ? v : v > 0 ? SHRT_MAX : SHRT_MIN); }
template<> inline mv_int16 saturate_cast<mv_int16>(mv_uint64 v)     { return (mv_int16)MIN(v, (mv_uint64)SHRT_MAX); }
template<> inline mv_int16 saturate_cast<mv_int16>(mv_float v)      { mv_int32 iv = mv_round(v); return saturate_cast<mv_int16>(iv); }
template<> inline mv_int16 saturate_cast<mv_int16>(mv_double v)     { mv_int32 iv = mv_round(v); return saturate_cast<mv_int16>(iv); }


template<> inline mv_uint16 saturate_cast<mv_uint16>(mv_int8 v)     { return (mv_uint16)MAX((mv_int32)v, 0); }
template<> inline mv_uint16 saturate_cast<mv_uint16>(mv_int16 v)    { return (mv_uint16)MAX((mv_int32)v, 0); }
template<> inline mv_uint16 saturate_cast<mv_uint16>(mv_int32 v)    { return (mv_uint16)((mv_uint32)v <= (mv_uint32)USHRT_MAX ? v : v > 0 ? USHRT_MAX : 0); }
template<> inline mv_uint16 saturate_cast<mv_uint16>(mv_uint32 v)   { return (mv_uint16)MIN(v, (mv_uint32)USHRT_MAX); }
template<> inline mv_uint16 saturate_cast<mv_uint16>(mv_int64 v)    { return (mv_uint16)((mv_uint64)v <= (mv_uint64)USHRT_MAX ? v : v > 0 ? USHRT_MAX : 0); }
template<> inline mv_uint16 saturate_cast<mv_uint16>(mv_uint64 v)   { return (mv_uint16)MIN(v, (mv_uint64)USHRT_MAX); }
template<> inline mv_uint16 saturate_cast<mv_uint16>(mv_float v)    { mv_int32 iv = mv_round(v); return saturate_cast<mv_uint16>(iv); }
template<> inline mv_uint16 saturate_cast<mv_uint16>(mv_double v)   { mv_int32 iv = mv_round(v); return saturate_cast<mv_uint16>(iv); }


template<> inline mv_int32 saturate_cast<mv_int32>(mv_float v)      { return mv_round(v); }
template<> inline mv_int32 saturate_cast<mv_int32>(mv_double v)     { return mv_round(v); }

// we intentionally do not clip negative numbers, to make -1 become 0xffffffff etc.
template<> inline mv_uint32 saturate_cast<mv_uint32>(mv_float v)    { return mv_round(v); }
template<> inline mv_uint32 saturate_cast<mv_uint32>(mv_double v)   { return mv_round(v); }









// 13  先不移植，等图像部分移植完成后再考虑



//
//
//
//double mv_invert(mv_mat_handle src, mv_mat_handle dst);  // MV_LU 高斯消去法   MV_SVD
//void mv_matrix_mul(const mv_mat_handle src1, const mv_mat_handle src2, mv_mat_handle dst);
//void mv_matrix_mul_add_ex(const mv_mat_handle src1, const mv_mat_handle src2, double alpha,
//    const mv_mat_handle src3, double beta, mv_mat_handle dst, int tABC);


///** Matrix transform: dst = A*B + C, C is optional */
//#define cvMatMulAdd( src1, src2, src3, dst ) cvGEMM( (src1), (src2), 1., (src3), 1., (dst), 0 )
//#define cvMatMul( src1, src2, dst )  cvMatMulAdd( (src1), (src2), NULL, (dst))
//
//#define CV_GEMM_A_T 1
//#define CV_GEMM_B_T 2
//#define CV_GEMM_C_T 4
///** Extended matrix transform:
//dst = alpha*op(A)*op(B) + beta*op(C), where op(X) is X or X^T */
//CVAPI(void)  cvGEMM(const CvArr* src1, const CvArr* src2, double alpha,
//    const CvArr* src3, double beta, CvArr* dst,
//    int tABC CV_DEFAULT(0));
//#define cvMatMulAddEx cvGEMM





//void mv_convert(const mv_matrix_t* src, mv_matrix_t* dst);     // COPY?
//void mv_copy(const mv_matrix_t* src, mv_matrix_t* dst, const mv_matrix_t* mask);
//void mv_eigen_val_vector(mv_matrix_t* mat, mv_matrix_t* evects, mv_matrix_t* evals, double eps, int lowindex, int highindex);  //OK
//void mv_svd(mv_matrix_t* A, mv_matrix_t* W, mv_matrix_t* U, mv_matrix_t* V, int flags);  //OK




void* mv_malloc(size_t size);
void mv_free(void* ptr);


#endif  //__MV_BASE_H__

