// mv_base.h write by scott.zgeng 2015.9.19
// 用于移植 opensift使用了opencv的部分


#ifndef  __MV_BASE_H__
#define  __MV_BASE_H__



#include <stdlib.h>


// fundamental constants 
#define MV_PI   3.1415926535897932384626433832795
#define MV_LOG2 0.69314718055994530941723212145818


#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif


#define CV_CN_MAX     512
#define CV_CN_SHIFT   3
#define CV_DEPTH_MAX  (1 << CV_CN_SHIFT)

#define CV_8U   0
#define CV_8S   1
#define CV_16U  2
#define CV_16S  3
#define CV_32S  4
#define CV_32F  5
#define CV_64F  6
#define CV_USRTYPE1 7

#define CV_MAT_DEPTH_MASK       (CV_DEPTH_MAX - 1)
#define CV_MAT_DEPTH(flags)     ((flags) & CV_MAT_DEPTH_MASK)

#define CV_MAKETYPE(depth,cn) (CV_MAT_DEPTH(depth) + (((cn)-1) << CV_CN_SHIFT))
#define CV_MAKE_TYPE CV_MAKETYPE

#define CV_8UC1 CV_MAKETYPE(CV_8U,1)
#define CV_8UC2 CV_MAKETYPE(CV_8U,2)
#define CV_8UC3 CV_MAKETYPE(CV_8U,3)
#define CV_8UC4 CV_MAKETYPE(CV_8U,4)
#define CV_8UC(n) CV_MAKETYPE(CV_8U,(n))

#define CV_8SC1 CV_MAKETYPE(CV_8S,1)
#define CV_8SC2 CV_MAKETYPE(CV_8S,2)
#define CV_8SC3 CV_MAKETYPE(CV_8S,3)
#define CV_8SC4 CV_MAKETYPE(CV_8S,4)
#define CV_8SC(n) CV_MAKETYPE(CV_8S,(n))

#define CV_16UC1 CV_MAKETYPE(CV_16U,1)
#define CV_16UC2 CV_MAKETYPE(CV_16U,2)
#define CV_16UC3 CV_MAKETYPE(CV_16U,3)
#define CV_16UC4 CV_MAKETYPE(CV_16U,4)
#define CV_16UC(n) CV_MAKETYPE(CV_16U,(n))

#define CV_16SC1 CV_MAKETYPE(CV_16S,1)
#define CV_16SC2 CV_MAKETYPE(CV_16S,2)
#define CV_16SC3 CV_MAKETYPE(CV_16S,3)
#define CV_16SC4 CV_MAKETYPE(CV_16S,4)
#define CV_16SC(n) CV_MAKETYPE(CV_16S,(n))

#define CV_32SC1 CV_MAKETYPE(CV_32S,1)
#define CV_32SC2 CV_MAKETYPE(CV_32S,2)
#define CV_32SC3 CV_MAKETYPE(CV_32S,3)
#define CV_32SC4 CV_MAKETYPE(CV_32S,4)
#define CV_32SC(n) CV_MAKETYPE(CV_32S,(n))

#define CV_32FC1 CV_MAKETYPE(CV_32F,1)
#define CV_32FC2 CV_MAKETYPE(CV_32F,2)
#define CV_32FC3 CV_MAKETYPE(CV_32F,3)
#define CV_32FC4 CV_MAKETYPE(CV_32F,4)
#define CV_32FC(n) CV_MAKETYPE(CV_32F,(n))

#define CV_64FC1 CV_MAKETYPE(CV_64F,1)
#define CV_64FC2 CV_MAKETYPE(CV_64F,2)
#define CV_64FC3 CV_MAKETYPE(CV_64F,3)
#define CV_64FC4 CV_MAKETYPE(CV_64F,4)
#define CV_64FC(n) CV_MAKETYPE(CV_64F,(n))

#define CV_MAT_CN_MASK          ((CV_CN_MAX - 1) << CV_CN_SHIFT)
#define CV_MAT_CN(flags)        ((((flags) & CV_MAT_CN_MASK) >> CV_CN_SHIFT) + 1)
#define CV_MAT_TYPE_MASK        (CV_DEPTH_MAX*CV_CN_MAX - 1)
#define CV_MAT_TYPE(flags)      ((flags) & CV_MAT_TYPE_MASK)
#define CV_MAT_CONT_FLAG_SHIFT  14
#define CV_MAT_CONT_FLAG        (1 << CV_MAT_CONT_FLAG_SHIFT)
#define CV_IS_MAT_CONT(flags)   ((flags) & CV_MAT_CONT_FLAG)
#define CV_IS_CONT_MAT          CV_IS_MAT_CONT
#define CV_SUBMAT_FLAG_SHIFT    15
#define CV_SUBMAT_FLAG          (1 << CV_SUBMAT_FLAG_SHIFT)


#define CV_MAGIC_MASK       0xFFFF0000
#define CV_MAT_MAGIC_VAL    0x42420000


/* 0x3a50 = 11 10 10 01 01 00 00 ~ array of log2(sizeof(arr_type_elem)) */
#define CV_ELEM_SIZE(type) \
    (CV_MAT_CN(type) << ((((sizeof(size_t) / 4 + 1) * 16384 | 0x3a50) >> CV_MAT_DEPTH(type) * 2) & 3))



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



#define CV_INTER_NN 0
#define CV_INTER_LINEAR  1
#define CV_INTER_CUBIC  2
#define CV_INTER_AREA  3
#define CV_INTER_LANCZOS4  4

#define CV_WARP_FILL_OUTLIERS  8
#define CV_WARP_INVERSE_MAP 16







/** linear convolution with \f$\texttt{size1}\times\texttt{size2}\f$ box kernel (all 1's). If
you want to smooth different pixels with different-size box kernels, you can use the integral
image that is computed using integral */
#define CV_BLUR_NO_SCALE 0
#define  CV_BLUR 1
#define  CV_GAUSSIAN 2
#define  CV_MEDIAN 3
#define  CV_BILATERAL 4


#define CV_WHOLE_SEQ_END_INDEX 0x3fffffff
#define CV_WHOLE_SEQ  mv_slice_t(0, CV_WHOLE_SEQ_END_INDEX)


#define CV_LU  0
#define CV_SVD 1
#define CV_SVD_SYM 2
#define CV_CHOLESKY 3
#define CV_QR  4
#define CV_NORMAL 16


#define CV_SVD_MODIFY_A   1
#define CV_SVD_U_T        2
#define CV_SVD_V_T        4


#define CV_GEMM_A_T 1
#define CV_GEMM_B_T 2
#define CV_GEMM_C_T 4


#define CV_AUTOSTEP  0x7fffffff


struct mv_roi_t
{
    int  coi; /**< 0 - no COI (all channels are selected), 1 - 0th channel is selected ...*/
    int  xOffset;
    int  yOffset;
    int  width;
    int  height;
};


struct mv_image_t
{
    int  nSize;             /**< sizeof(IplImage) */
    int  ID;                /**< version (=0)*/
    int  nChannels;         /**< Most of OpenCV functions support 1,2,3 or 4 channels */
    int  alphaChannel;      /**< Ignored by OpenCV */
    int  depth;             /**< Pixel depth in bits: IPL_DEPTH_8U, IPL_DEPTH_8S, IPL_DEPTH_16S,
                            IPL_DEPTH_32S, IPL_DEPTH_32F and IPL_DEPTH_64F are supported.  */
    char colorModel[4];     /**< Ignored by OpenCV */
    char channelSeq[4];     /**< ditto */
    int  dataOrder;         /**< 0 - interleaved color channels, 1 - separate color channels.
                            cvCreateImage can only create interleaved images */
    int  origin;            /**< 0 - top-left origin,
                            1 - bottom-left origin (Windows bitmaps style).  */
    int  align;             /**< Alignment of image rows (4 or 8).
                            OpenCV ignores it and uses widthStep instead.    */
    int  width;             /**< Image width in pixels.                           */
    int  height;            /**< Image height in pixels.                          */
    
    mv_roi_t *roi;    /**< Image ROI. If NULL, the whole image is selected. */
    mv_image_t *maskROI;      /**< Must be NULL. */
    
    void  *imageId;                 /**< "           " */
    //struct _IplTileInfo *tileInfo;  /**< "           " */  
    int  imageSize;         /**< Image data size in bytes
                            (==image->height*image->widthStep
                            in case of interleaved data)*/
    char *imageData;        /**< Pointer to aligned image data.         */
    int  widthStep;         /**< Size of aligned image row in bytes.    */
    int  BorderMode[4];     /**< Ignored by OpenCV.                     */
    int  BorderConst[4];    /**< Ditto.                                 */
    char *imageDataOrigin;  /**< Pointer to very origin of image data
                            (not necessarily aligned) -
                            needed for correct deallocation */

};


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

struct mv_matrix_t
{
    int type;
    int step;

    // for internal use only 
    int* refcount;
    int hdr_refcount;

    union
    {
        unsigned char* ptr;
        short* s;
        int* i;
        float* fl;
        double* db;
    } data;

    int rows;
    int cols;


    mv_matrix_t() {
        type = 0;
        type = 0;
        cols = 0;
        rows = 0;
        step = 0;
        data.ptr = NULL;
        refcount = NULL;
        hdr_refcount = 0;
    }

    mv_matrix_t(int r, int c, int t, void* ptr)
    {        
        type = CV_MAT_TYPE(t);
        type = CV_MAT_MAGIC_VAL | CV_MAT_CONT_FLAG | t;
        cols = c;
        rows = r;
        step = c * CV_ELEM_SIZE(t);
        data.ptr = (unsigned char*)ptr;
        refcount = NULL;
        hdr_refcount = 0;        
    }
};







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
#elif ((defined _MSC_VER && defined _M_ARM) || defined CV_ICC || \
    defined __GNUC__) && defined HAVE_TEGRA_OPTIMIZATION
    TEGRA_ROUND_DBL(value);
#elif defined CV_ICC || defined __GNUC__
# if CV_VFP
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


inline void  mv_matrix_set(mv_matrix_t* mat, int row, int col, double value)
{
    int type;
    type = CV_MAT_TYPE(mat->type);
    //assert((unsigned)row < (unsigned)mat->rows && (unsigned)col < (unsigned)mat->cols);

    if (type == CV_32FC1)
        ((float*)(void*)(mat->data.ptr + (size_t)mat->step*row))[col] = (float)value;
    else
    {
        //assert(type == CV_64FC1);
        ((double*)(void*)(mat->data.ptr + (size_t)mat->step*row))[col] = value;
    }
}



// 15
void mv_set_zero(mv_image_t* arr);
mv_image_t* mv_create_image(mv_size_t size, int depth, int channels);
mv_image_t* mv_clone_image(const mv_image_t* image);
void mv_release_image(mv_image_t** image);
void mv_set_image_roi(mv_image_t* image, mv_rect_t rect);
void mv_reset_image_roi(mv_image_t* image);
void mv_add_weighted(const mv_image_t* src1, double alpha, const mv_image_t* src2, double beta, double gamma, mv_image_t* dst);
void mv_add(const mv_image_t* src1, const mv_image_t* src2, mv_image_t* dst, const mv_image_t* mask);
void mv_sub(const mv_image_t* src1, const mv_image_t* src2, mv_image_t* dst, const mv_image_t* mask);
void mv_resize(const mv_image_t* src, mv_image_t* dst, int interpolation);
mv_size_t mv_get_size(const mv_image_t* arr);
void mv_cvt_color(const mv_image_t* src, mv_image_t* dst, int code);
void mv_convert_scale(const mv_image_t* src, mv_image_t* dst, double scale, double shift);
void mv_smooth(const mv_image_t* src, mv_image_t* dst, int smoothtype, int size1, int size2, double sigma1, double sigma2);
void mv_warp_perspective(const mv_image_t* src, mv_image_t* dst, const mv_matrix_t* map_matrix, int flags, mv_scalar_t fillval);



// 13
mv_matrix_t* mv_create_matrix(int rows, int cols, int type);
double mv_invert(mv_matrix_t* src, mv_matrix_t* dst, int method);
void mv_release_matrix(mv_matrix_t** mat);
mv_matrix_t* mv_clone_matrix(const mv_matrix_t* mat);
mv_matrix_t* mv_init_matrix_header(mv_matrix_t* mat, int rows, int cols, int type, void* data, int step);
void mv_matrix_mul_add_ex(const mv_matrix_t* src1, const mv_matrix_t* src2, double alpha,
    const mv_matrix_t* src3, double beta, mv_matrix_t* dst, int tABC);
#define mv_matrix_mul(src1, src2, dst)  mv_matrix_mul_add_ex((src1), (src2), 1.0, NULL, 1.0, (dst), 0)
void mv_eigen_val_vector(mv_matrix_t* mat, mv_matrix_t* evects, mv_matrix_t* evals, double eps, int lowindex, int highindex);
void mv_matrix_zero(mv_matrix_t* mat);
void mv_svd(mv_matrix_t* A, mv_matrix_t* W, mv_matrix_t* U, mv_matrix_t* V, int flags);
mv_matrix_t* mv_get_row(const mv_matrix_t* arr, mv_matrix_t* submat, int row);
void mv_convert(const mv_matrix_t* src, mv_matrix_t* dst);
void mv_copy(const mv_matrix_t* src, mv_matrix_t* dst, const mv_matrix_t* mask);
int mv_solve(const mv_matrix_t* src1, const mv_matrix_t* src2, mv_matrix_t* dst, int method);




// TODO(scott.zgeng): 画图函数，后续是不需要的
void mv_ellipse(mv_image_t* img, mv_point_t center, mv_size_t axes,
    double angle, double start_angle, double end_angle, mv_scalar_t color, int thickness, int line_type, int shift);
void mv_line(mv_image_t* img, mv_point_t pt1, mv_point_t pt2, mv_scalar_t color, int thickness, int line_type, int shift);
int mv_wait_key(int key);
void mv_named_window(const char* name);
void mv_show_image(const char* name, const mv_image_t* image);
void mv_destroy_window(const char* name);
void* mv_get_windows_handle(const char* name);
mv_image_t* mv_load_image(const char* filename, int iscolor);



#endif  //__MV_BASE_H__

