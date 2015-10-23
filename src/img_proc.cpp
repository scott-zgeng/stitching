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


void mv_resize_cubic(const mv_image_t* src, mv_image_t* dst)
{

    double scale_x = (double)src->width / dst->width;
    double scale_y = (double)src->height / dst->height;
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
    double scale_x = (double)src->width / dst->width;
    double scale_y = (double)src->height / dst->height;

    for (int y = 0; y < dst->height; ++y) {

        int sy = mv_floor(y * scale_y);
        sy = MIN(sy, src->height - 1);

        for (int x = 0; x < dst->width; ++x) {
            int sx = mv_floor(x * scale_x);
            sx = MIN(sx, src->width - 1);
            dst->imageData[y * dst->widthStep + x] = src->imageData[sy * src->widthStep + sx];
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

        for (int i = 0; i < 256; i++, b += db, g += dg, r += dr) {
            blue_tab[i] = b;
            green_tab[i] = g;
            red_tab[i] = r;
        }
    }
    void operator ()(const mv_byte* src, mv_byte* dst, int n) const
    {
        int scn = srccn;
        
        for (int i = 0; i < n; i++, src += scn)
            dst[i] = (mv_byte)((blue_tab[src[0]] + green_tab[src[1]] + red_tab[src[2]]) >> yuv_shift);

            
    }
    int srccn;
    int blue_tab[256];
    int green_tab[256];
    int red_tab[256];
};


void mv_convert_gray(const mv_image_t* src, mv_image_t* dst)
{
    rgb2gray cvt(3, 0);
    cvt(src->imageData, dst->imageData, src->width * src->height);
}



void mv_warp_perspective(const mv_image_t* src, mv_image_t* dst, const mv_matrix_t* map_matrix, int flags, mv_scalar_t fillval)
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


/** dst(mask) = src1(mask) - src2(mask) */
void mv_sub(const mv_image_t* src1, const mv_image_t* src2, mv_image_t* dst)
{    
    assert(src1->width == src2->width && src1->height == src2->height);
    assert(src1->width == dst->width && src1->height == dst->height);

    mv_byte* p1 = src1->imageData;
    mv_byte* p2 = src2->imageData;
    mv_byte* result = dst->imageData;

    int size = src1->height * src1->width;
    
    for (int i = 0; i < size; i++) {
        result[i] = saturate_cast<mv_byte>(p1[i] - p2[i]);
    }
}



// box blur 
// refencence： http://blog.ivank.net/fastest-gaussian-blur.html
static void box_blur_horizontal(mv_byte* src, mv_byte* dst, int width, int height, double sigma)
{
    double iarr = 1 / (sigma + sigma + 1);

    for (int i = 0; i<height; i++) {
        int ti = i*width;
        int li = ti;
        int ri = ti + sigma;
        double fv = src[ti];
        double lv = src[ti + width - 1];
        double val = (sigma + 1)*fv;

        for (int j = 0; j < sigma; j++) {
            val += src[ti + j];
        }

        for (int j = 0; j <= sigma; j++) { 
            val += src[ri++] - fv;   
            dst[ti++] = mv_round(val*iarr);
        }

        for (int j = sigma + 1; j<width - sigma; j++) {
            val += src[ri++] - src[li++];   
            dst[ti++] = mv_round(val*iarr);
        }

        for (int j = width - sigma; j<width; j++) { 
            val += lv - src[li++];   
            dst[ti++] = mv_round(val*iarr);
        }
    }
}

static void box_blur_total(mv_byte* src, mv_byte* dst, int width, int height, double sigma)
{
    double iarr = 1 / (sigma + sigma + 1);

    for (int i = 0; i < width; i++) {
        int ti = i;
        int li = ti;
        int ri = ti + sigma*width;
        double fv = src[ti];
        double lv = src[ti + width*(height - 1)];
        double val = (sigma + 1)*fv;

        for (int j = 0; j < sigma; j++) {
            val += src[ti + j*width];
        }

        for (int j = 0; j <= sigma; j++) { 
            val += src[ri] - fv;  
            dst[ti] = mv_round(val*iarr);  
            ri += width; ti += width; 
        }

        for (int j = sigma + 1; j < height - sigma; j++) { 
            val += src[ri] - src[li];  
            dst[ti] = mv_round(val*iarr);
            li += width; ri += width; ti += width; 
        }

        for (int j = height - sigma; j < height; j++) {
            val += lv - src[li];  
            dst[ti] = mv_round(val*iarr);
            li += width; ti += width; 
        }
    }
}


static inline void box_blur_impl(mv_byte* src, mv_byte* dst, int width, int height, double sigma)
{
    int size = width * height;
    for (int i = 0; i<size; i++)
        dst[i] = src[i];

    box_blur_horizontal(dst, src, width, height, sigma);
    box_blur_total(src, dst, width, height, sigma);
}

// todo(scott.zgeng): 缓冲后续最好是在外部申请
void mv_box_blur(const mv_image_t* src, mv_image_t* dst, double sigma)
{
    static const int PASS_NUM = 3;
    double kernel[PASS_NUM];

    double ideal_width  = sqrt((12 * sigma*sigma / PASS_NUM) + 1);  // Ideal averaging filter width 
    int wl = mv_floor(ideal_width);  if (wl % 2 == 0) wl--;
    int wu = wl + 2;

    double mIdeal = (12 * sigma*sigma - PASS_NUM*wl*wl - 4 * PASS_NUM*wl - 3*PASS_NUM) / (-4*wl - 4);
    int m = mv_round(mIdeal);
    //double sigma_actual = sqrt( (m*wl*wl + (n-m)*wu*wu - n)/12 );

    for (int i = 0; i < PASS_NUM; i++) {
        kernel[i] = ((i < m ? wl : wu) - 1) / 2.0;        
    }

    int width = src->width;
    int height = src->height;

    mv_byte* p1 = (mv_byte* )mv_malloc(width*height);
    memcpy(p1, src->imageData, width*height);
    mv_byte* p2 = dst->imageData;

    box_blur_impl(p1, p2, width, height, kernel[0]);
    box_blur_impl(p2, p1, width, height, kernel[1]);
    box_blur_impl(p1, p2, width, height, kernel[2]);

    mv_free(p1);
}

