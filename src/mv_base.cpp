// mv_base.cpp write by scott.zgeng 2015.9.19
// 用于移植 opensift使用了opencv的部分

#include "mv_base.h"

#include <cxcore.h>
#include <cv.h>
#include <highgui.h>


void* mv_malloc(size_t size)
{
    return malloc(size);
}

void mv_free(void* ptr)
{
    free(ptr);
}


mv_image_t* mv_load_image(const char* filename, int iscolor)
{
    IplImage* org = cvLoadImage(filename, iscolor);
    mv_image_t* image = (mv_image_t*)malloc(sizeof(mv_image_t));

    image->org = org;

    image->nSize = sizeof(mv_image_t);      // sizeof(IplImage) 
    image->ID = org->ID;                    // version (=0)
    image->nChannels = org->nChannels;      // Most of OpenCV functions support 1,2,3 or 4 channels 
    //image->alphaChannel = org->alphaChannel;      // Ignored by OpenCV 
    image->depth = org->depth;              // Pixel depth in bits
    //image->colorModel[4] = image->colorModel;     // Ignored by OpenCV 
    //image->channelSeq[0] = image->channelSeq[0];  // ditto 

    image->dataOrder = org->dataOrder;      // 0 - interleaved color channels, 1 - separate color channels. cvCreateImage can only create interleaved images
    image->origin = org->origin;            // top-left origin, 1 - bottom-left origin (Windows bitmaps style).  
    image->align = org->align;              // Alignment of image rows (4 or 8).OpenCV ignores it and uses widthStep instead. 
    image->width = org->width;              // Image width in pixels.                           
    image->height = org->height;            // Image height in pixels.                         

    image->roi = NULL;                      // Image ROI. If NULL, the whole image is selected. 
    image->maskROI = NULL;                  // Must be NULL. 

    image->imageId = org->imageId;
    //struct _IplTileInfo *tileInfo;        //
    image->imageSize = org->imageSize;      // Image data size in bytes                            
    image->imageData = (mv_byte*)org->imageData;      // Pointer to aligned image data.         
    image->widthStep = org->widthStep;      // Size of aligned image row in bytes.    
    //image->BorderMode[4];                 // Ignored by OpenCV.                     
    //image->BorderConst[4] = image->BorderConst[0]; // Ditto.                   
    image->imageDataOrigin = (mv_byte*)org->imageDataOrigin;  // Pointer to very origin of image data

    return image;
}



static void icvGetColorModel(int nchannels, const char** colorModel, const char** channelSeq)
{
    static const char* tab[][2] = {
        { "GRAY", "GRAY" },
        { "", "" },
        { "RGB", "BGR" },
        { "RGB", "BGRA" }
    };

    nchannels--;
    *colorModel = *channelSeq = "";

    if ((unsigned)nchannels <= 3) {
        *colorModel = tab[nchannels][0];
        *channelSeq = tab[nchannels][1];
    }
}



int mv_init_image_header(mv_image_t* image, mv_size_t size, int depth, int channels, int origin, int align)
{
    const char *colorModel, *channelSeq;

    if (!image) {
        return MV_FAILED;
    }

    image->org = NULL;
    memset(image, 0, sizeof(mv_image_t));
    image->nSize = sizeof(mv_image_t);

    icvGetColorModel(channels, &colorModel, &channelSeq);
    strncpy(image->colorModel, colorModel, 4);
    strncpy(image->channelSeq, channelSeq, 4);

    if (size.width < 0 || size.height < 0) {
        return MV_FAILED;
    }

    //if ((depth != (int)IPL_DEPTH_1U && depth != (int)IPL_DEPTH_8U &&
    //    depth != (int)IPL_DEPTH_8S && depth != (int)IPL_DEPTH_16U &&
    //    depth != (int)IPL_DEPTH_16S && depth != (int)IPL_DEPTH_32S &&
    //    depth != (int)IPL_DEPTH_32F && depth != (int)IPL_DEPTH_64F) ||
    //    channels < 0)
    //    CV_Error(CV_BadDepth, "Unsupported format");
    //if (origin != 0 && origin != 1)
    //    CV_Error(CV_BadOrigin, "Bad input origin");

    //if (align != 4 && align != 8)
    //    CV_Error(CV_BadAlign, "Bad input align");

    image->width = size.width;
    image->height = size.height;

    if (image->roi) {
        image->roi->coi = 0;
        image->roi->xOffset = image->roi->yOffset = 0;
        image->roi->width = size.width;
        image->roi->height = size.height;
    }

    image->nChannels = MAX(channels, 1);
    image->depth = depth;
    image->align = align;

    image->widthStep = (((image->width * image->nChannels * (image->depth & ~IPL_DEPTH_SIGN) + 7) / 8) + align - 1) & (~(align - 1));
    image->origin = origin;
    image->imageSize = image->widthStep * image->height;

    return MV_SUCCEEDED;
}



#define IPL_ORIGIN_TL 0
#define IPL_ORIGIN_BL 1

mv_image_t* mv_create_image(mv_size_t size, int depth, int channels)
{
    mv_image_t* img = (mv_image_t *)mv_malloc(sizeof(mv_image_t));
    mv_init_image_header(img, size, depth, channels, IPL_ORIGIN_TL, 4);

    img->imageDataOrigin = (mv_byte*)mv_malloc((size_t)img->imageSize);
    img->imageData = img->imageDataOrigin;

    return img;
}



mv_image_t* mv_clone_image(const mv_image_t* src)
{
    mv_image_t* dst = (mv_image_t*)mv_malloc(sizeof(mv_image_t));

    memcpy(dst, src, sizeof(mv_image_t));
    dst->imageData = dst->imageDataOrigin = NULL;
    dst->roi = NULL;

    //if (src->roi) {    
    //    dst->roi = icvCreateROI(src->roi->coi, src->roi->xOffset, src->roi->yOffset, src->roi->width, src->roi->height);
    //}

    if (src->imageData) {
        int size = src->imageSize;

        dst->imageDataOrigin = (mv_byte*)mv_malloc((size_t)size);
        dst->imageData = dst->imageDataOrigin;
        memcpy(dst->imageData, src->imageData, size);
    }

    return dst;
}


class rgb2gray
{
    static const int R2Y = 4899;
    static const int G2Y = 9617;
    static const int B2Y = 1868;
    static const int yuv_shift = 14;

public:
    rgb2gray(int _srccn, int blueIdx) : srccn(_srccn)
    {
        const int coeffs[] = { R2Y, G2Y, B2Y };

        int b = 0, g = 0, r = (1 << (yuv_shift - 1));
        int db = coeffs[blueIdx ^ 2], dg = coeffs[1], dr = coeffs[blueIdx];

        for (int i = 0; i < 256; i++, b += db, g += dg, r += dr)
        {
            tab[i] = b;
            tab[i + 256] = g;
            tab[i + 512] = r;
        }
    }
    void operator ()(const mv_byte* src, mv_byte* dst, int n) const
    {
        int scn = srccn;
        const int* _tab = tab;
        for (int i = 0; i < n; i++, src += scn)
            dst[i] = (mv_byte)((_tab[src[0]] + _tab[src[1] + 256] + _tab[src[2] + 512]) >> yuv_shift);
    }
    int srccn;
    int tab[256 * 3];
};


void mv_cvt_bgr_gray(const mv_image_t* src, mv_image_t* dst)
{
    rgb2gray cvt(3, 0);
    cvt(src->imageData, dst->imageData, src->width * src->height);
}


void mv_release_image(mv_image_t** image)
{

}

void mv_set_image_roi(mv_image_t* image, mv_rect_t rect)
{

}


void mv_reset_image_roi(mv_image_t* image)
{

}


void mv_add_weighted(const mv_image_t* src1, double alpha, const mv_image_t* src2, double beta, double gamma, mv_image_t* dst)
{

}

// dst(mask) = src1(mask) + src2(mask) 
void mv_add(const mv_image_t* src1, const mv_image_t* src2, mv_image_t* dst, const mv_image_t* mask)
{

}

/** dst(mask) = src1(mask) - src2(mask) */
void mv_sub(const mv_image_t* src1, const mv_image_t* src2, mv_image_t* dst, const mv_image_t* mask)
{

}




template<typename T, typename WT, typename AT>
struct HResizeCubic
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const T** src, WT** dst, int count,
        const int* xofs, const AT* alpha,
        int swidth, int dwidth, int cn, int xmin, int xmax) const
    {
        for (int k = 0; k < count; k++)
        {
            const T *S = src[k];
            WT *D = dst[k];
            int dx = 0, limit = xmin;
            for (;;)
            {
                for (; dx < limit; dx++, alpha += 4)
                {
                    int j, sx = xofs[dx] - cn;
                    WT v = 0;
                    for (j = 0; j < 4; j++)
                    {
                        int sxj = sx + j*cn;
                        if ((unsigned)sxj >= (unsigned)swidth)
                        {
                            while (sxj < 0)
                                sxj += cn;
                            while (sxj >= swidth)
                                sxj -= cn;
                        }
                        v += S[sxj] * alpha[j];
                    }
                    D[dx] = v;
                }
                if (limit == dwidth)
                    break;
                for (; dx < xmax; dx++, alpha += 4)
                {
                    int sx = xofs[dx];
                    D[dx] = S[sx - cn] * alpha[0] + S[sx] * alpha[1] +
                        S[sx + cn] * alpha[2] + S[sx + cn * 2] * alpha[3];
                }
                limit = dwidth;
            }
            alpha -= dwidth * 4;
        }
    }
};


template<typename ST, typename DT, int bits> struct FixedPtCast
{
    typedef ST type1;
    typedef DT rtype;
    enum { SHIFT = bits, DELTA = 1 << (bits - 1) };

    DT operator()(ST val) const { return saturate_cast<DT>((val + DELTA) >> SHIFT); }
};



template<typename T, typename WT, typename AT, class CastOp>
struct VResizeCubic
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const WT** src, T* dst, const AT* beta, int width) const
    {
        WT b0 = beta[0], b1 = beta[1], b2 = beta[2], b3 = beta[3];
        const WT *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
        CastOp castOp;

        for (int x = 0; x < width; x++)
            dst[x] = castOp(S0[x] * b0 + S1[x] * b1 + S2[x] * b2 + S3[x] * b3);
    }
};





mv_result resize_cubic::init(mv_size_t ssize, mv_size_t dsize) {

    m_ssize = ssize;
    m_dsize = dsize;

    double inv_scale_x = (double)dsize.width / ssize.width;
    double inv_scale_y = (double)dsize.height / ssize.height;

    int cn = 1;  //channel number

    double scale_x = 1.0 / inv_scale_x;
    double scale_y = 1.0 / inv_scale_y;

    m_xmin = 0;
    m_xmax = dsize.width;        

    m_ksize = 4;
    int ksize2 = m_ksize / 2;

    m_xofs = (int*)mv_malloc(dsize.width*sizeof(int));
    m_yofs = (int*)mv_malloc(dsize.height*sizeof(int));
    m_alpha = (short*)mv_malloc(dsize.width*m_ksize*sizeof(short));
    m_beta = (short*)mv_malloc(dsize.height*m_ksize*sizeof(short));
        

    int k, sx, sy, dx, dy;
    float fx, fy;
    for (int dx = 0; dx < dsize.width; dx++) {
        fx = (float)((dx + 0.5)*scale_x - 0.5);
        sx = mv_floor(fx);
        fx -= sx;

        if (sx < ksize2 - 1)
            m_xmin = dx + 1;

        if (sx + ksize2 >= ssize.width)
            m_xmax = MAX(m_xmax, dx);

        for (k = 0, sx *= cn; k < cn; k++)
            m_xofs[dx*cn + k] = sx + k;

        interpolate_cubic(fx, m_cbuf);

        for (k = 0; k < m_ksize; k++)
            m_alpha[dx*cn*m_ksize + k] = saturate_cast<short>(m_cbuf[k] * INTER_RESIZE_COEF_SCALE);

        for (; k < cn*m_ksize; k++)
            m_alpha[dx*cn*m_ksize + k] = m_alpha[dx*cn*m_ksize + k - m_ksize];

    }

    for (dy = 0; dy < dsize.height; dy++) {
        fy = (float)((dy + 0.5)*scale_y - 0.5);
        sy = mv_floor(fy);
        fy -= sy;

        m_yofs[dy] = sy;

        interpolate_cubic(fy, m_cbuf);

        for (k = 0; k < m_ksize; k++)
            m_beta[dy*m_ksize + k] = saturate_cast<short>(m_cbuf[k] * INTER_RESIZE_COEF_SCALE);
    }

    return MV_SUCCEEDED;
}

static inline int clip(int x, int a, int b)
{
    return x >= a ? (x < b ? x : b - 1) : a;
}


static inline size_t align_size(size_t sz, int n)
{    
    return (sz + n - 1) & -n;
}

void resize_cubic::operator()(const mv_image_t* src, mv_image_t* dst) 
{
    HResizeCubic<uchar, int, short> hresize;
    VResizeCubic<uchar, int, short, FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS * 2> > vresize;

    int cn = src->nChannels;

    
    int bufstep = (int)align_size(m_dsize.width, 16);
    AutoBuffer<int> _buffer(bufstep*m_ksize);

    const uchar* srows[MAX_ESIZE] = { 0 };
    int* rows[MAX_ESIZE] = { 0 };
    int prev_sy[MAX_ESIZE];

    for (int k = 0; k < m_ksize; k++)
    {
        prev_sy[k] = -1;
        rows[k] = (int*)_buffer + bufstep*k;
    }


    int ksize2 = m_ksize / 2;
    const short* beta = m_beta;


    for (int dy = 0; dy < m_dsize.height; dy++, beta += m_ksize)
    {
        int sy0 = m_yofs[dy], k0 = m_ksize, k1 = 0;

        for (int k = 0; k < m_ksize; k++)
        {
            int sy = clip(sy0 - ksize2 + 1 + k, 0, m_ssize.height);
            for (int k1 = MAX(k1, k); k1 < m_ksize; k1++)
            {
                if (sy == prev_sy[k1]) // if the sy-th row has been computed already, reuse it.
                {
                    if (k1 > k)
                        memcpy(rows[k], rows[k1], bufstep*sizeof(rows[0][0]));
                    break;
                }
            }
            if (k1 == m_ksize)
                k0 = std::min(k0, k); // remember the first row that needs to be computed
            //srows[k] = src->imageData; .template ptr<uchar>(sy);
            prev_sy[k] = sy;
        }

        if (k0 < m_ksize) {
            hresize((const uchar**)(srows + k0), (int**)(rows + k0), m_ksize - k0, m_xofs,
                m_alpha, m_ssize.width, m_dsize.width, cn, m_xmin, m_xmax);
        }

        vresize((const int**)rows, (uchar*)(dst->imageData + dst->widthStep*dy), beta, m_dsize.width);
    }


}


void mv_resize_nn(const mv_image_t* src, mv_image_t* dst)
{

}

mv_size_t mv_get_size(const mv_image_t* image)
{
    return mv_size_t(image->width, image->height);
}



void mv_normalize_u8(const mv_image_t* src, mv_image_t* dst, double scale)
{
    int  width = src->width;
    int  height = src->height;
    mv_byte* src_data = src->imageData;
    float* dst_data = (float*)dst->imageData;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            dst_data[x] = (float)(src_data[x] * scale);
        }
        src_data += src->width;
        dst_data += dst->width;
    }
}






void mv_smooth(const mv_image_t* src, mv_image_t* dst, int smoothtype, int size1, int size2, double sigma1, double sigma2)
{

}



int mv_wait_key(int key)
{
    return 0;
}




void mv_named_window(const char* name)
{

}

void mv_show_image(const char* name, const mv_image_t* image)
{

}

mv_matrix_t* mv_create_matrix(int rows, int cols, int type)
{
    return NULL;
}

double mv_invert(mv_matrix_t* src, mv_matrix_t* dst, int method)
{
    return 0.0;
}

void mv_release_matrix(mv_matrix_t** mat)
{

}

mv_matrix_t* mv_clone_matrix(const mv_matrix_t* mat)
{
    return NULL;

}

mv_matrix_t* mv_init_matrix_header(mv_matrix_t* mat, int rows, int cols, int type, void* data, int step)
{
    return NULL;
}


void mv_matrix_mul_add_ex(const mv_matrix_t* src1, const mv_matrix_t* src2, double alpha,
    const mv_matrix_t* src3, double beta, mv_matrix_t* dst, int tABC)
{

}


void mv_warp_perspective(const mv_image_t* src, mv_image_t* dst, const mv_matrix_t* map_matrix, int flags, mv_scalar_t fillval)
{

}




void mv_eigen_val_vector(mv_matrix_t* mat, mv_matrix_t* evects, mv_matrix_t* evals, double eps, int lowindex, int highindex)
{
}


void mv_matrix_zero(mv_matrix_t* mat)
{

}


// Performs Singular Value Decomposition of a matrix 
void mv_svd(mv_matrix_t* A, mv_matrix_t* W, mv_matrix_t* U, mv_matrix_t* V, int flags)
{

}



mv_matrix_t* mv_get_row(const mv_matrix_t* arr, mv_matrix_t* submat, int row)
{
    return NULL;
}


void mv_convert(const mv_matrix_t* src, mv_matrix_t* dst)
{

}



void mv_copy(const mv_matrix_t* src, mv_matrix_t* dst, const mv_matrix_t* mask)
{

}

/** Solves linear system (src1)*(dst) = (src2)
(returns 0 if src1 is a singular and MV_LU method is used) */
int mv_solve(const mv_matrix_t* src1, const mv_matrix_t* src2, mv_matrix_t* dst, int method)
{
    return 0;
}


// TODO(scott.zgeng): 画图函数，后续是不需要的
void mv_ellipse(mv_image_t* img, mv_point_t center, mv_size_t axes,
    double angle, double start_angle, double end_angle, mv_scalar_t color, int thickness, int line_type, int shift)
{

}

void mv_line(mv_image_t* img, mv_point_t pt1, mv_point_t pt2, mv_scalar_t color, int thickness, int line_type, int shift)
{
    //return cvLine(img, pt1, pt2, color, thickness, line_type, shift);
}




void mv_set_zero(mv_image_t* arr)
{
}

void mv_destroy_window(const char* name)
{

}

void* mv_get_windows_handle(const char* name)
{
    return NULL;
}

