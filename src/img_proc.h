// img_proc.h write by scott.zgeng 2015.10.13


#ifndef  __IMG_PROC_H__
#define  __IMG_PROC_H__


#include "mv_base.h"



struct mv_roi_t
{
    int coi; // 0 - no COI (all channels are selected), 1 - 0th channel is selected ...
    int xOffset;
    int yOffset;
    int width;
    int height;
};


struct mv_image_t
{
    int nSize;                  // sizeof(IplImage) 
    int ID;                     // version (=0)
    int nChannels;              // Most of OpenCV functions support 1,2,3 or 4 channels     
    int depth;                  // Pixel depth in bits: IPL_DEPTH_8U, IPL_DEPTH_8S, IPL_DEPTH_16S,
                                // IPL_DEPTH_32S, IPL_DEPTH_32F and IPL_DEPTH_64F are supported. 
    int dataOrder;              // 0 - interleaved color channels, 1 - separate color channels.
                                // cvCreateImage can only create interleaved images 
    int origin;                 // 0 - top-left origin,
                                // 1 - bottom-left origin (Windows bitmaps style).     
    int width;                  // Image width in pixels.
    int height;                 // Image height in pixels.    
    mv_roi_t* roi;              // Image ROI. If NULL, the whole image is selected.    
    void* imageId;
    int imageSize;              // Image data size in bytes (==image->height*image->widthStep in case of interleaved data)
    int widthStep;              // Size of aligned image row in bytes.     
    mv_byte* imageData;         // Pointer to aligned image data.          
    mv_byte* imageDataOrigin;   // Pointer to very origin of image data (not necessarily aligned) - needed for correct deallocation
};




//---------------  FOR TEST ---------------
// notes(scott.zgeng): 图像转换函数主要是方便测试，正式环境需要剔除
#include "highgui.h"
mv_image_t* mv_image_cv2mv(IplImage* org);  // OK
IplImage* mv_image_mv2cv(mv_image_t* org);  // OK
//---------------  FOR TEST END -----------


// FOR SIFT
static inline float pixval32f(mv_image_t* img, int r, int c)
{
    return ((float*)(img->imageData + img->widthStep*r))[c];
}


// 已经实现的函数
void mv_convert_gray(const mv_image_t* src, mv_image_t* dst);   // OK
mv_image_t* mv_clone_image(const mv_image_t* image);  // OK
mv_image_t* mv_create_image(mv_size_t size, int depth, int channels);
void mv_release_image(mv_image_t** image);
void mv_normalize_u8(const mv_image_t* src, mv_image_t* dst, double scale); // [OK]


static inline mv_size_t mv_get_size(const mv_image_t* image) {
    return mv_size_t(image->width, image->height);
}


// todo(scott.zgeng): 未整理
void mv_set_zero(mv_image_t* arr);

void mv_set_image_roi(mv_image_t* image, mv_rect_t rect);
void mv_reset_image_roi(mv_image_t* image);
void mv_add_weighted(const mv_image_t* src1, double alpha, const mv_image_t* src2, double beta, double gamma, mv_image_t* dst);
void mv_add(const mv_image_t* src1, const mv_image_t* src2, mv_image_t* dst, const mv_image_t* mask);
void mv_sub(const mv_image_t* src1, const mv_image_t* src2, mv_image_t* dst, const mv_image_t* mask);

void mv_resize_cubic(const mv_image_t* src, mv_image_t* dst);
void mv_resize_nn(const mv_image_t* src, mv_image_t* dst);

void mv_smooth(const mv_image_t* src, mv_image_t* dst, int smoothtype, int size1, int size2, double sigma1, double sigma2);
void mv_warp_perspective(const mv_image_t* src, mv_image_t* dst, const mv_matrix_t* map_matrix, int flags, mv_scalar_t fillval);




#endif //__IMG_PROC_H__

