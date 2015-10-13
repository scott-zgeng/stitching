// img_proc.cpp write by scott.zgeng 2015.10.13


#include <assert.h>

#include "mv_base.h"
#include "img_proc.h"



mv_image_t* mv_image_cv2mv(IplImage* org)
{
    mv_image_t* image = (mv_image_t*)mv_malloc(sizeof(mv_image_t));

    image->nSize = sizeof(mv_image_t);
    image->ID = org->ID;
    image->nChannels = org->nChannels;    
    image->depth = org->depth;    
    image->dataOrder = org->dataOrder;
    image->origin = org->origin;    
    image->width = org->width;
    image->height = org->height;
    image->roi = NULL;   
    image->imageId = org->imageId;    
    image->imageSize = org->imageSize;    
    image->widthStep = org->widthStep;
    image->imageData = (mv_byte*)mv_malloc(org->imageSize);
    memcpy(image->imageData, org->imageData, org->imageSize);
    image->imageDataOrigin = image->imageDataOrigin;

    return image;
}


IplImage* mv_image_mv2cv(mv_image_t* org)
{    
    IplImage* image = cvCreateImage(cvSize(org->width, org->height), org->depth, org->nChannels);
    if (NULL == image)
        return NULL;

    memcpy(image->imageData, org->imageData, org->imageSize);
    return image;
}



int mv_init_image_header(mv_image_t* image, mv_size_t size, int depth, int channels, int origin, int align)
{
    if (!image) 
        return MV_FAILED;    

    if (size.width < 0 || size.height < 0) 
        return MV_FAILED;
    
    memset(image, 0, sizeof(mv_image_t));
    image->nSize = sizeof(mv_image_t);

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
    assert(src->roi == NULL);

    mv_image_t* dst = (mv_image_t*)mv_malloc(sizeof(mv_image_t));

    memcpy(dst, src, sizeof(mv_image_t));
    dst->imageData = dst->imageDataOrigin = NULL;
    dst->roi = NULL;

    if (src->imageData) {
        int size = src->imageSize;

        dst->imageDataOrigin = (mv_byte*)mv_malloc((size_t)size);
        dst->imageData = dst->imageDataOrigin;
        memcpy(dst->imageData, src->imageData, size);
    }

    return dst;
}


void mv_release_image(mv_image_t** image)
{
    mv_image_t* p = *image;

    if (p != NULL) {
        mv_free(p);
    }

    *image = NULL;   
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


//
//
//template<typename T, typename WT, typename AT>
//struct HResizeCubic
//{
//    typedef T value_type;
//    typedef WT buf_type;
//    typedef AT alpha_type;
//
//    void operator()(const T** src, WT** dst, int count,
//        const int* xofs, const AT* alpha,
//        int swidth, int dwidth, int cn, int xmin, int xmax) const
//    {
//        for (int k = 0; k < count; k++)
//        {
//            const T *S = src[k];
//            WT *D = dst[k];
//            int dx = 0, limit = xmin;
//            for (;;)
//            {
//                for (; dx < limit; dx++, alpha += 4)
//                {
//                    int j, sx = xofs[dx] - cn;
//                    WT v = 0;
//                    for (j = 0; j < 4; j++)
//                    {
//                        int sxj = sx + j*cn;
//                        if ((unsigned)sxj >= (unsigned)swidth)
//                        {
//                            while (sxj < 0)
//                                sxj += cn;
//                            while (sxj >= swidth)
//                                sxj -= cn;
//                        }
//                        v += S[sxj] * alpha[j];
//                    }
//                    D[dx] = v;
//                }
//                if (limit == dwidth)
//                    break;
//                for (; dx < xmax; dx++, alpha += 4)
//                {
//                    int sx = xofs[dx];
//                    D[dx] = S[sx - cn] * alpha[0] + S[sx] * alpha[1] +
//                        S[sx + cn] * alpha[2] + S[sx + cn * 2] * alpha[3];
//                }
//                limit = dwidth;
//            }
//            alpha -= dwidth * 4;
//        }
//    }
//};
//
//
//template<typename ST, typename DT, int bits> struct FixedPtCast
//{
//    typedef ST type1;
//    typedef DT rtype;
//    enum { SHIFT = bits, DELTA = 1 << (bits - 1) };
//
//    DT operator()(ST val) const { return saturate_cast<DT>((val + DELTA) >> SHIFT); }
//};
//
//
//
//template<typename T, typename WT, typename AT, class CastOp>
//struct VResizeCubic
//{
//    typedef T value_type;
//    typedef WT buf_type;
//    typedef AT alpha_type;
//
//    void operator()(const WT** src, T* dst, const AT* beta, int width) const
//    {
//        WT b0 = beta[0], b1 = beta[1], b2 = beta[2], b3 = beta[3];
//        const WT *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
//        CastOp castOp;
//
//        for (int x = 0; x < width; x++)
//            dst[x] = castOp(S0[x] * b0 + S1[x] * b1 + S2[x] * b2 + S3[x] * b3);
//    }
//};
//


//
//
//mv_result resize_cubic::init(mv_size_t ssize, mv_size_t dsize) {
//
//    m_ssize = ssize;
//    m_dsize = dsize;
//
//    double inv_scale_x = (double)dsize.width / ssize.width;
//    double inv_scale_y = (double)dsize.height / ssize.height;
//
//    int cn = 1;  //channel number
//
//    m_scale_x = 1.0 / inv_scale_x;
//    m_scale_y = 1.0 / inv_scale_y;
//
//    m_xmin = 0;
//    m_xmax = dsize.width;        
//
//    m_ksize = 4;
//    int ksize2 = m_ksize / 2;
//
//    m_xofs = (int*)mv_malloc(dsize.width*sizeof(int));
//    m_yofs = (int*)mv_malloc(dsize.height*sizeof(int));
//    m_alpha = (short*)mv_malloc(dsize.width*m_ksize*sizeof(short));
//    m_beta = (short*)mv_malloc(dsize.height*m_ksize*sizeof(short));
//        
//
//    int k, sx, sy, dx, dy;
//    float fx, fy;
//    for (int dx = 0; dx < dsize.width; dx++) {
//        fx = (float)((dx + 0.5)*m_scale_x - 0.5);
//        sx = mv_floor(fx);
//        fx -= sx;
//
//        if (sx < ksize2 - 1)
//            m_xmin = dx + 1;
//
//        if (sx + ksize2 >= ssize.width)
//            m_xmax = MAX(m_xmax, dx);
//
//        for (k = 0, sx *= cn; k < cn; k++)
//            m_xofs[dx*cn + k] = sx + k;
//
//        interpolate_cubic(fx, m_cbuf);
//
//        for (k = 0; k < m_ksize; k++)
//            m_alpha[dx*cn*m_ksize + k] = saturate_cast<short>(m_cbuf[k] * INTER_RESIZE_COEF_SCALE);
//
//        for (; k < cn*m_ksize; k++)
//            m_alpha[dx*cn*m_ksize + k] = m_alpha[dx*cn*m_ksize + k - m_ksize];
//
//    }
//
//    for (dy = 0; dy < dsize.height; dy++) {
//        fy = (float)((dy + 0.5)*m_scale_y - 0.5);
//        sy = mv_floor(fy);
//        fy -= sy;
//
//        m_yofs[dy] = sy;
//
//        interpolate_cubic(fy, m_cbuf);
//
//        for (k = 0; k < m_ksize; k++)
//            m_beta[dy*m_ksize + k] = saturate_cast<short>(m_cbuf[k] * INTER_RESIZE_COEF_SCALE);
//    }
//
//    return MV_SUCCEEDED;
//}

//static inline int clip(int x, int a, int b)
//{
//    return x >= a ? (x < b ? x : b - 1) : a;
//}
//
//
//static inline size_t align_size(size_t sz, int n)
//{    
//    return (sz + n - 1) & -n;
//}
//
//

//void resize_cubic::operator()(const mv_image_t* src, mv_image_t* dst) 
//{
//    HResizeCubic<uchar, int, short> hresize;
//    VResizeCubic<uchar, int, short, FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS * 2> > vresize;
//
//    int cn = src->nChannels;
//
//    
//    int bufstep = (int)align_size(m_dsize.width, 16);
//    AutoBuffer<int> _buffer(bufstep*m_ksize);
//
//    const uchar* srows[MAX_ESIZE] = { 0 };
//    int* rows[MAX_ESIZE] = { 0 };
//    int prev_sy[MAX_ESIZE];
//
//    for (int k = 0; k < m_ksize; k++)
//    {
//        prev_sy[k] = -1;
//        rows[k] = (int*)_buffer + bufstep*k;
//    }
//
//    int ksize2 = m_ksize / 2;
//    const short* beta = m_beta;
//
//
//    for (int dy = 0; dy < m_dsize.height; dy++, beta += m_ksize)
//    {
//        int sy0 = m_yofs[dy], k0 = m_ksize, k1 = 0;
//
//        for (int k = 0; k < m_ksize; k++)
//        {
//            int sy = clip(sy0 - ksize2 + 1 + k, 0, m_ssize.height);
//            for (int k1 = MAX(k1, k); k1 < m_ksize; k1++)
//            {
//                if (sy == prev_sy[k1]) // if the sy-th row has been computed already, reuse it.
//                {
//                    if (k1 > k)
//                        memcpy(rows[k], rows[k1], bufstep*sizeof(rows[0][0]));
//                    break;
//                }
//            }
//            if (k1 == m_ksize)
//                k0 = MIN(k0, k); // remember the first row that needs to be computed
//            //srows[k] = src->imageData; .template ptr<uchar>(sy);
//            prev_sy[k] = sy;
//        }
//
//        if (k0 < m_ksize) {
//            hresize((const uchar**)(srows + k0), (int**)(rows + k0), m_ksize - k0, m_xofs,
//                m_alpha, m_ssize.width, m_dsize.width, cn, m_xmin, m_xmax);
//        }
//
//        vresize((const int**)rows, (uchar*)(dst->imageData + dst->widthStep*dy), beta, m_dsize.width);
//    }
//
//
//}

//
//void dasdasd()
//{
//    function gaussBlur_4(scl, tcl, w, h, r) {
//    
//    var bxs = boxesForGauss(r, 3);
//    
//    boxBlur_4(scl, tcl, w, h, (bxs[0] - 1) / 2);
//    
//            boxBlur_4(tcl, scl, w, h, (bxs[1] - 1) / 2);
//        
//            boxBlur_4(scl, tcl, w, h, (bxs[2] - 1) / 2);
//                
//    }
//    12.
//        13.function boxBlur_4(scl, tcl, w, h, r) {
//            14.
//                15.    for (var i = 0; i<scl.length; i++) tcl[i] = scl[i];
//            16.
//                17.    boxBlurH_4(tcl, scl, w, h, r);
//            18.
//                19.    boxBlurT_4(scl, tcl, w, h, r);
//            20.
//                21.
//        }
//    22.
//        23.function boxBlurH_4(scl, tcl, w, h, r) {
//            24.
//                25.    var iarr = 1 / (r + r + 1);
//            26.
//                27.    for (var i = 0; i<h; i++) {
//                28.
//                    29.        var ti = i*w, li = ti, ri = ti + r;
//                30.
//                    31.        var fv = scl[ti], lv = scl[ti + w - 1], val = (r + 1)*fv;
//                32.
//                    33.        for (var j = 0; j<r; j++) val += scl[ti + j];
//                34.
//                    35.        for (var j = 0; j <= r; j++) { val += scl[ri++] - fv;   tcl[ti++] = Math.round(val*iarr); }
//                36.
//                    37.        for (var j = r + 1; j<w - r; j++) { val += scl[ri++] - scl[li++];   tcl[ti++] = Math.round(val*iarr); }
//                38.
//                    39.        for (var j = w - r; j<w; j++) { val += lv - scl[li++];   tcl[ti++] = Math.round(val*iarr); }
//                40.
//                    41.
//            }
//            42.
//                43.
//        }
//    44.
//        45.function boxBlurT_4(scl, tcl, w, h, r) {
//            46.
//                47.    var iarr = 1 / (r + r + 1);
//            48.
//                49.    for (var i = 0; i<w; i++) {
//                50.
//                    51.        var ti = i, li = ti, ri = ti + r*w;
//                52.
//                    53.        var fv = scl[ti], lv = scl[ti + w*(h - 1)], val = (r + 1)*fv;
//                54.
//                    55.        for (var j = 0; j<r; j++) val += scl[ti + j*w];
//                56.
//                    57.        for (var j = 0; j <= r; j++) { val += scl[ri] - fv;  tcl[ti] = Math.round(val*iarr);  ri += w; ti += w; }
//                58.
//                    59.        for (var j = r + 1; j<h - r; j++) { val += scl[ri] - scl[li];  tcl[ti] = Math.round(val*iarr);  li += w; ri += w; ti += w; }
//                60.
//                    61.        for (var j = h - r; j<h; j++) { val += lv - scl[li];  tcl[ti] = Math.round(val*iarr);  li += w; ti += w; }
//                62.
//                    63.
//            }
//            64.
//                65.
//        }
//
//
//}



void mv_resize_cubic(const mv_image_t* src, mv_image_t* dst)
{

    double scale_x = src->width / dst->width;
    double scale_y = src->height / dst->height;
    uchar* dataDst = dst->imageData;
    int stepDst = dst->widthStep;
    uchar* dataSrc = src->imageData;
    int stepSrc = src->widthStep;
    int iWidthSrc = src->width;
    int iHiehgtSrc = src->height;

    for (int j = 0; j < dst->height; ++j)
    {
        float fy = (float)((j + 0.5) * scale_y - 0.5);
        int sy = mv_floor(fy);
        fy -= sy;
        sy = MIN(sy, iHiehgtSrc - 2);
        sy = MAX(0, sy);

        short cbufy[2];
        cbufy[0] = saturate_cast<short>((1.f - fy) * 2048);
        cbufy[1] = 2048 - cbufy[0];

        for (int i = 0; i < dst->width; ++i)
        {
            float fx = (float)((i + 0.5) * scale_x - 0.5);
            int sx = mv_floor(fx);
            fx -= sx;

            if (sx < 0) {
                fx = 0, sx = 0;
            }
            if (sx >= iWidthSrc - 1) {
                fx = 0, sx = iWidthSrc - 2;
            }

            short cbufx[2];
            cbufx[0] = saturate_cast<short>((1.f - fx) * 2048);
            cbufx[1] = 2048 - cbufx[0];

            dataDst[j*stepDst + i] =
                (
                dataSrc[sy*stepSrc + sx] * cbufx[0] * cbufy[0] +
                dataSrc[(sy + 1)*stepSrc + sx] * cbufx[0] * cbufy[1] +
                dataSrc[sy*stepSrc + (sx + 1)] * cbufx[1] * cbufy[0] +
                dataSrc[(sy + 1)*stepSrc + (sx + 1)] * cbufx[1] * cbufy[1]
                ) >> 22;
        }
    }

}

void mv_resize_nn(const mv_image_t* src, mv_image_t* dst)
{
    //int cn = 1;  //channel number

    double scale_x = src->width / dst->width;
    double scale_y = src->height / dst->height;

    for (int y = 0; y < dst->height; ++y) {

        int sy = mv_floor(y * scale_y);
        sy = MIN(sy, src->height - 1);

        for (int x = 0; x < dst->width; ++x) {
            int sx = mv_floor(x * scale_x);
            sx = MIN(sx, src->width - 1);
            dst->imageData[y * dst->widthStep, x] = src->imageData[sy * src->widthStep + sx];
        }
    }
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


void mv_convert_gray(const mv_image_t* src, mv_image_t* dst)
{
    rgb2gray cvt(3, 0);
    cvt(src->imageData, dst->imageData, src->width * src->height);
}



void mv_warp_perspective(const mv_image_t* src, mv_image_t* dst, const mv_matrix_t* map_matrix, int flags, mv_scalar_t fillval)
{

}



void mv_set_zero(mv_image_t* arr)
{
}
