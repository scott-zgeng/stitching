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
