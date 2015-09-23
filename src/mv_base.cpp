// mv_base.cpp write by scott.zgeng 2015.9.19
// 用于移植 opensift使用了opencv的部分

#include "mv_base.h"

//#include <cxcore.h>
//#include <cv.h>
//#include <highgui.h>


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

mv_image_t* mv_load_image(const char* filename, int iscolor)
{
    return NULL;

}

mv_image_t* mv_create_image(mv_size_t size, int depth, int channels)
{
    return NULL;
}


mv_image_t* mv_clone_image(const mv_image_t* image)
{
    return NULL;
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


void mv_resize(const mv_image_t* src, mv_image_t* dst, int interpolation)
{

}

mv_size_t mv_get_size(const mv_image_t* arr)
{
    return mv_size_t();
}



void mv_cvt_color(const mv_image_t* src, mv_image_t* dst, int code)
{

}


void mv_convert_scale(const mv_image_t* src, mv_image_t* dst, double scale, double shift)
{

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
(returns 0 if src1 is a singular and CV_LU method is used) */
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


