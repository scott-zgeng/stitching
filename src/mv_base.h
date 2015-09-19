// mv_base.h write by scott.zgeng 2015.9.19
// 用于移植 opensift使用了opencv的部分


#ifndef  __MV_BASE_H__
#define  __MV_BASE_H__

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


#define MV_RGB( r, g, b )  mv_scalar_t( (b), (g), (r), 0 )

struct mv_point_t
{
    int x;
    int y;

    mv_point_t(int _x, int _y) {
        x = _x;
        y = _y;
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

    mv_matrix_t(int r, int c, int t, void* ptr)
    {
        
        type = CV_MAT_TYPE(t);
        type = CV_MAT_MAGIC_VAL | CV_MAT_CONT_FLAG | t;
        cols = c;
        rows = r;
        step = c * CV_ELEM_SIZE(t);
        data.ptr = (uchar*)ptr;
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


mv_image_t* mv_load_image(const char* filename, int iscolor);

void mv_line(mv_image_t* img, mv_point_t pt1, mv_point_t pt2, mv_scalar_t color, int thickness, int line_type, int shift);

void mv_wait_key(int key);

#endif  //__MV_BASE_H__

