/*
  Functions for detecting SIFT image features.

  For more information, refer to:

  Lowe, D.  Distinctive image features from scale-invariant keypoints.
  <EM>International Journal of Computer Vision, 60</EM>, 2 (2004),
  pp.91--110.

  Copyright (C) 2006-2012  Rob Hess <rob@iqengines.com>

  Note: The SIFT algorithm is patented in the United States and cannot be
  used in commercial products without a license from the University of
  British Columbia.  For more information, refer to the file LICENSE.ubc
  that accompanied this distribution.

  @version 1.1.2-20100521
  */


#include <assert.h>
#include <math.h>
#include <string.h>

#include "sift.h"
#include "imgfeatures.h"
#include "utils.h"

#include "../img_proc.h"
#include "../mat_proc.h"



static void show_temp_image(mv_image_t* img)
{
    const char* IMG_TEST_1 = "test_windows_1";
    IplImage* clone = mv_image_mv2cv(img);
    cvShowImage(IMG_TEST_1, clone);
    cvWaitKey(0);
    cvReleaseImage(&clone);
}


/************************* Local Function Prototypes *************************/

static mv_image_t* create_init_img(mv_image_t*, int, double);
static mv_image_t* convert_to_gray32(mv_image_t*);
static mv_image_t*** build_gauss_pyr(mv_image_t*, int, int, double);
static mv_image_t* downsample(mv_image_t*);
static mv_image_t*** build_dog_pyr(mv_image_t***, int, int);
static int scale_space_extrema(mv_image_t***, int, int, double, int, mv_features* features);

static int is_extremum(mv_image_t***, int, int, int, int);
static struct feature* interp_extremum(mv_image_t***, int, int, int, int, int,
    double);
static void interp_step(mv_image_t***, int, int, int, int, double*, double*,
    double*);
static void deriv_3D(mv_image_t*** dog_pyr, int octv, int intvl, int r, int c, double* result);
static mv_mat_handle hessian_3D(mv_image_t***, int, int, int, int);
static double interp_contr(mv_image_t***, int, int, int, int, double, double,
    double);
static struct feature* new_feature(void);
static int is_too_edge_like(mv_image_t*, int, int, int);
static void calc_feature_scales(mv_features*, double, int);
static void adjust_for_img_dbl(mv_features*);
static void calc_feature_oris(mv_features*, mv_image_t***);
static double* ori_hist(mv_image_t*, int, int, int, int, double);
static int calc_grad_mag_ori(mv_image_t*, int, int, double*, double*);
static void smooth_ori_hist(double*, int);
static double dominant_ori(double*, int);
static void add_good_ori_features(mv_features*, double*, int, double,
struct feature*);
static struct feature* clone_feature(struct feature*);
static void compute_descriptors(mv_features*, mv_image_t***, int, int);
static double*** descr_hist(mv_image_t*, int, int, double, double, int, int);
static void interp_hist_entry(double***, double, double, double, double, int,
    int);
static void hist_to_descr(double***, int, int, struct feature*);
static void normalize_descr(struct feature*);
static int feature_cmp(void*, void*, void*);
static void release_descr_hist(double****, int);
static void release_pyr(mv_image_t****, int, int);


/*********************** Functions prototyped in sift.h **********************/





/************************ Functions prototyped here **************************/

/*
  Converts an image to 8-bit grayscale and Gaussian-smooths it.  The image is
  optionally doubled in size prior to smoothing.

  @param img input image
  @param img_dbl if true, image is doubled in size prior to smoothing
  @param sigma total std of Gaussian smoothing
  */

#include "cv.h"



/*
  Converts an image to 32-bit grayscale

  @param img a 3-channel 8-bit color (BGR) or 8-bit gray image

  @return Returns a 32-bit grayscale image
  */



/*
  Builds Gaussian scale space pyramid from an image

  @param base base image of the pyramid
  @param octvs number of octaves of scale space
  @param intvls number of intervals per octave
  @param sigma amount of Gaussian smoothing per octave

  @return Returns a Gaussian scale space pyramid as an octvs x (intvls + 3)
  array
  */
int sift_runtime::build_gauss_pyramid(mv_image_t* base, int octvs, double sigma)
{    
    double sig[SIFT_INTVLS + 3];
    double sig_total, sig_prev, k;
    int i, o;

    //gauss_pyr = (mv_image_t***)calloc(octvs, sizeof(mv_image_t**));
    //for (i = 0; i < octvs; i++)
        //gauss_pyr[i] = (mv_image_t **)calloc(SIFT_INTVLS + 3, sizeof(mv_image_t *));

    /*
      precompute Gaussian sigmas using the following formula:

      \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2

      sig[i] is the incremental sigma value needed to compute
      the actual sigma of level i. Keeping track of incremental
      sigmas vs. total sigmas keeps the gaussian kernel small.
      */
    double k = pow(2.0, 1.0 / SIFT_INTVLS);
    sig[0] = sigma;
    sig[1] = sigma * sqrt(k*k - 1);
    for (i = 2; i < SIFT_INTVLS + 3; i++)
        sig[i] = sig[i - 1] * k;

    for (int o = 0; o < octvs; o++)
    for (int i = 0; i < SIFT_INTVLS + 3; i++)
    {        
        if (o == 0 && i == 0)
            set_gauss_pyramid(mv_clone_image(base), o, i);

        /* base of new octvave is halved image from end of previous octave */
        else if (i == 0)
            set_gauss_pyramid(downsample(gauss_pyramid(o - 1, SIFT_INTVLS)), o, i);

        /* blur the current octave's last image to create the next one */
        else
        {
            mv_image_t* prev_layer = gauss_pyramid(o, i - 1);
            set_gauss_pyramid(mv_create_image(mv_get_size(prev_layer), IPL_DEPTH_32F, 1), o, i);
            mv_box_blur(prev_layer, gauss_pyramid(o, i), sig[i]);
        }
    }

    return 0;
}



/*
  Downsamples an image to a quarter of its size (half in each dimension)
  using nearest-neighbor interpolation

  @param img an image

  @return Returns an image whose dimensions are half those of img
  */
mv_image_t* sift_runtime::downsample(mv_image_t* img)
{
    mv_image_t* smaller = mv_create_image(mv_size_t(img->width / 2, img->height / 2),
        img->depth, img->nChannels);
    mv_resize_nn(img, smaller);

    return smaller;
}



/*
  Builds a difference of Gaussians scale space pyramid by subtracting adjacent
  intervals of a Gaussian pyramid

  @param gauss_pyr Gaussian scale-space pyramid
  @param octvs number of octaves of scale space
  @param intvls number of intervals per octave

  @return Returns a difference of Gaussians scale space pyramid as an
  octvs x (intvls + 2) array
  */
int sift_runtime::build_dog_pyr(int octvs, int intvls)
{    

    for (int o = 0; o < octvs; o++) {
        for (int i = 0; i < intvls + 2; i++) {
            mv_image_t* prev = gauss_pyramid(o, i + 1);
            mv_image_t* curr = gauss_pyramid(o, i);
            mv_image_t* dog = mv_create_image(mv_get_size(), IPL_DEPTH_32F, 1);
            set_dog_pyramid(dog, o, i);
            mv_sub(prev, curr, dog);

            WRITE_INFO_LOG("SUB o = %d i = %d", o, i);
            //show_temp_image(dog_pyr[o][i]);
        }
    }

    return 0;
}



/*
  Detects features at extrema in DoG scale space.  Bad features are discarded
  based on contrast and ratio of principal curvatures.

  @param dog_pyr DoG scale space pyramid
  @param octvs octaves of scale space represented by dog_pyr
  @param intvls intervals per octave
  @param contr_thr low threshold on feature contrast
  @param curv_thr high threshold on feature ratio of principal curvatures
  @param storage memory storage in which to store detected features

  @return Returns an array of detected features whose scales, orientations,
  and descriptors are yet to be determined.
  */
int sift_runtime::scale_space_extrema(mv_image_t*** dog_pyr, int octvs, int intvls,
    double contr_thr, int curv_thr, mv_features* features)
{
    double prelim_contr_thr = 0.5 * contr_thr / intvls;
    struct feature* feat;
    struct detection_data* ddata;
    int o, i, r, c;

    for (o = 0; o < octvs; o++) {
        for (i = 1; i <= intvls; i++) {
            for (r = SIFT_IMG_BORDER; r < dog_pyr[o][0]->height - SIFT_IMG_BORDER; r++) {
                for (c = SIFT_IMG_BORDER; c < dog_pyr[o][0]->width - SIFT_IMG_BORDER; c++) {
                    
                    /* perform preliminary check on contrast */
                    if (ABS(pixval32f(dog_pyr[o][i], r, c)) > prelim_contr_thr) {
                        if (is_extremum(dog_pyr, o, i, r, c))
                        {
                            feat = interp_extremum(dog_pyr, o, i, r, c, intvls, contr_thr);
                            if (feat)
                            {
                                ddata = feat_detection_data(feat);
                                if (!is_too_edge_like(dog_pyr[ddata->octv][ddata->intvl],
                                    ddata->r, ddata->c, curv_thr)) 
                                {
                                    WRITE_INFO_LOG("keypoint o=%d, i=%d, r=%d, c=%d", o, i, r, c);
                                    features->push_back(feat);
                                    
                                } else {
                                    free(ddata);
                                    free(feat); // 此处代码有修改，
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}



/*
  Determines whether a pixel is a scale-space extremum by comparing it to it's
  3x3x3 pixel neighborhood.

  @param dog_pyr DoG scale space pyramid
  @param octv pixel's scale space octave
  @param intvl pixel's within-octave interval
  @param r pixel's image row
  @param c pixel's image col

  @return Returns 1 if the specified pixel is an extremum (max or min) among
  it's 3x3x3 pixel neighborhood.
  */
static int sift_runtime::is_extremum(mv_image_t*** dog_pyr, int octv, int intvl, int r, int c)
{
    double val = pixval32f(dog_pyr[octv][intvl], r, c);
    int i, j, k;

    /* check for maximum */
    if (val > 0)
    {
        for (i = -1; i <= 1; i++)
        for (j = -1; j <= 1; j++)
        for (k = -1; k <= 1; k++)
        if (val < pixval32f(dog_pyr[octv][intvl + i], r + j, c + k))
            return 0;
    }

    /* check for minimum */
    else
    {
        for (i = -1; i <= 1; i++)
        for (j = -1; j <= 1; j++)
        for (k = -1; k <= 1; k++)
        if (val > pixval32f(dog_pyr[octv][intvl + i], r + j, c + k))
            return 0;
    }

    return 1;
}



/*
  Interpolates a scale-space extremum's location and scale to subpixel
  accuracy to form an image feature.  Rejects features with low contrast.
  Based on Section 4 of Lowe's paper.

  @param dog_pyr DoG scale space pyramid
  @param octv feature's octave of scale space
  @param intvl feature's within-octave interval
  @param r feature's image row
  @param c feature's image column
  @param intvls total intervals per octave
  @param contr_thr threshold on feature contrast

  @return Returns the feature resulting from interpolation of the given
  parameters or NULL if the given location could not be interpolated or
  if contrast at the interpolated loation was too low.  If a feature is
  returned, its scale, orientation, and descriptor are yet to be determined.
  */
static struct feature* sift_runtime::interp_extremum(mv_image_t*** dog_pyr, int octv,
    int intvl, int r, int c, int intvls,
    double contr_thr)
{
    struct feature* feat;
    struct detection_data* ddata;
    double xi, xr, xc, contr;
    int i = 0;

    while (i < SIFT_MAX_INTERP_STEPS)
    {
        interp_step(dog_pyr, octv, intvl, r, c, &xi, &xr, &xc);
        if (ABS(xi) < 0.5  &&  ABS(xr) < 0.5  &&  ABS(xc) < 0.5)
            break;

        c += mv_round(xc);
        r += mv_round(xr);
        intvl += mv_round(xi);

        if (intvl < 1 ||
            intvl > intvls ||
            c < SIFT_IMG_BORDER ||
            r < SIFT_IMG_BORDER ||
            c >= dog_pyr[octv][0]->width - SIFT_IMG_BORDER ||
            r >= dog_pyr[octv][0]->height - SIFT_IMG_BORDER)
        {
            return NULL;
        }

        i++;
    }

    /* ensure convergence of interpolation */
    if (i >= SIFT_MAX_INTERP_STEPS)
        return NULL;

    contr = interp_contr(dog_pyr, octv, intvl, r, c, xi, xr, xc);
    if (ABS(contr) < contr_thr / intvls)
        return NULL;

    feat = new_feature();
    ddata = feat_detection_data(feat);
    feat->img_pt.x = feat->x = (c + xc) * pow(2.0, octv);
    feat->img_pt.y = feat->y = (r + xr) * pow(2.0, octv);
    ddata->r = r;
    ddata->c = c;
    ddata->octv = octv;
    ddata->intvl = intvl;
    ddata->subintvl = xi;

    return feat;
}



/*
  Performs one step of extremum interpolation.  Based on Eqn. (3) in Lowe's
  paper.

  @param dog_pyr difference of Gaussians scale space pyramid
  @param octv octave of scale space
  @param intvl interval being interpolated
  @param r row being interpolated
  @param c column being interpolated
  @param xi output as interpolated subpixel increment to interval
  @param xr output as interpolated subpixel increment to row
  @param xc output as interpolated subpixel increment to col
  */

static void sift_runtime::interp_step(mv_image_t*** dog_pyr, int octv, int intvl, int r, int c,
    double* xi, double* xr, double* xc)
{

    double deriv_v[3];
    deriv_3D(dog_pyr, octv, intvl, r, c, deriv_v);
    mv_mat_handle dD = mv_create_matrix(3, 1);
    mv_matrix_set(dD, 0, 0, -deriv_v[0]);
    mv_matrix_set(dD, 1, 0, -deriv_v[1]);
    mv_matrix_set(dD, 2, 0, -deriv_v[2]);

    mv_mat_handle H = hessian_3D(dog_pyr, octv, intvl, r, c);
    mv_mat_handle H_inv = mv_create_matrix(3, 3);

    mv_invert_svd(H, H_inv);

    mv_mat_handle X = mv_create_matrix(3, 1);// , MV_64FC1, x, MV_AUTOSTEP);
    mv_matrix_mul(H_inv, dD, X);

    mv_release_matrix(dD);
    mv_release_matrix(H);
    mv_release_matrix(H_inv);

    *xi = mv_matrix_get(X, 2, 0);
    *xr = mv_matrix_get(X, 1, 0);
    *xc = mv_matrix_get(X, 0, 0);
}


/*
  Computes the partial derivatives in x, y, and scale of a pixel in the DoG
  scale space pyramid.

  @param dog_pyr DoG scale space pyramid
  @param octv pixel's octave in dog_pyr
  @param intvl pixel's interval in octv
  @param r pixel's image row
  @param c pixel's image col

  @return Returns the vector of partial derivatives for pixel I
  { dI/dx, dI/dy, dI/ds }^T as a CvMat*
  */
static void sift_runtime::deriv_3D(mv_image_t*** dog_pyr, int octv, int intvl, int r, int c, double* result)
{
    double dx = (pixval32f(dog_pyr[octv][intvl], r, c + 1) -
        pixval32f(dog_pyr[octv][intvl], r, c - 1)) / 2.0;
    double dy = (pixval32f(dog_pyr[octv][intvl], r + 1, c) -
        pixval32f(dog_pyr[octv][intvl], r - 1, c)) / 2.0;
    double ds = (pixval32f(dog_pyr[octv][intvl + 1], r, c) -
        pixval32f(dog_pyr[octv][intvl - 1], r, c)) / 2.0;

    result[0] = dx;
    result[1] = dy;
    result[2] = ds;

    return;
}



/*
  Computes the 3D Hessian matrix for a pixel in the DoG scale space pyramid.

  @param dog_pyr DoG scale space pyramid
  @param octv pixel's octave in dog_pyr
  @param intvl pixel's interval in octv
  @param r pixel's image row
  @param c pixel's image col

  @return Returns the Hessian matrix (below) for pixel I as a CvMat*

  / Ixx  Ixy  Ixs \ <BR>
  | Ixy  Iyy  Iys | <BR>
  \ Ixs  Iys  Iss /
  */
static mv_mat_handle sift_runtime::hessian_3D(mv_image_t*** dog_pyr, int octv, int intvl, int r,
    int c)
{
    mv_mat_handle H;
    double v, dxx, dyy, dss, dxy, dxs, dys;

    v = pixval32f(dog_pyr[octv][intvl], r, c);
    dxx = (pixval32f(dog_pyr[octv][intvl], r, c + 1) +
        pixval32f(dog_pyr[octv][intvl], r, c - 1) - 2 * v);
    dyy = (pixval32f(dog_pyr[octv][intvl], r + 1, c) +
        pixval32f(dog_pyr[octv][intvl], r - 1, c) - 2 * v);
    dss = (pixval32f(dog_pyr[octv][intvl + 1], r, c) +
        pixval32f(dog_pyr[octv][intvl - 1], r, c) - 2 * v);
    dxy = (pixval32f(dog_pyr[octv][intvl], r + 1, c + 1) -
        pixval32f(dog_pyr[octv][intvl], r + 1, c - 1) -
        pixval32f(dog_pyr[octv][intvl], r - 1, c + 1) +
        pixval32f(dog_pyr[octv][intvl], r - 1, c - 1)) / 4.0;
    dxs = (pixval32f(dog_pyr[octv][intvl + 1], r, c + 1) -
        pixval32f(dog_pyr[octv][intvl + 1], r, c - 1) -
        pixval32f(dog_pyr[octv][intvl - 1], r, c + 1) +
        pixval32f(dog_pyr[octv][intvl - 1], r, c - 1)) / 4.0;
    dys = (pixval32f(dog_pyr[octv][intvl + 1], r + 1, c) -
        pixval32f(dog_pyr[octv][intvl + 1], r - 1, c) -
        pixval32f(dog_pyr[octv][intvl - 1], r + 1, c) +
        pixval32f(dog_pyr[octv][intvl - 1], r - 1, c)) / 4.0;

    H = mv_create_matrix(3, 3);
    mv_matrix_set(H, 0, 0, dxx);
    mv_matrix_set(H, 0, 1, dxy);
    mv_matrix_set(H, 0, 2, dxs);
    mv_matrix_set(H, 1, 0, dxy);
    mv_matrix_set(H, 1, 1, dyy);
    mv_matrix_set(H, 1, 2, dys);
    mv_matrix_set(H, 2, 0, dxs);
    mv_matrix_set(H, 2, 1, dys);
    mv_matrix_set(H, 2, 2, dss);

    return H;
}



/*
  Calculates interpolated pixel contrast.  Based on Eqn. (3) in Lowe's
  paper.

  @param dog_pyr difference of Gaussians scale space pyramid
  @param octv octave of scale space
  @param intvl within-octave interval
  @param r pixel row
  @param c pixel column
  @param xi interpolated subpixel increment to interval
  @param xr interpolated subpixel increment to row
  @param xc interpolated subpixel increment to col

  @param Returns interpolated contrast.
  */
static double sift_runtime::interp_contr(mv_image_t*** dog_pyr, int octv, int intvl, int r,
    int c, double xi, double xr, double xc)
{
    mv_mat_handle X = mv_create_matrix(3, 1);
    mv_matrix_set(X, 0, 0, xc);
    mv_matrix_set(X, 0, 1, xr);
    mv_matrix_set(X, 0, 2, xi);

    mv_mat_handle T = mv_create_matrix(1, 1);

    double deriv_x[3];
    deriv_3D(dog_pyr, octv, intvl, r, c, deriv_x);
    mv_mat_handle dD = mv_create_matrix(1, 3);
    mv_matrix_set(dD, 0, 0, deriv_x[0]);
    mv_matrix_set(dD, 0, 1, deriv_x[1]);
    mv_matrix_set(dD, 0, 2, deriv_x[2]);
    mv_matrix_mul(dD, X, T);

    mv_release_matrix(dD);

    double t = mv_matrix_get(T, 0, 0);

    return pixval32f(dog_pyr[octv][intvl], r, c) + t * 0.5;
}



/*
  Allocates and initializes a new feature

  @return Returns a pointer to the new feature
  */
static struct feature* sift_runtime::new_feature(void)
{
    struct feature* feat;
    struct detection_data* ddata;

    feat = (struct feature*)malloc(sizeof(struct feature));
    memset(feat, 0, sizeof(struct feature));
    ddata = (struct detection_data*)malloc(sizeof(struct detection_data));
    memset(ddata, 0, sizeof(struct detection_data));
    feat->feature_data = ddata;
    feat->type = FEATURE_LOWE;

    return feat;
}



/*
  Determines whether a feature is too edge like to be stable by computing the
  ratio of principal curvatures at that feature.  Based on Section 4.1 of
  Lowe's paper.

  @param dog_img image from the DoG pyramid in which feature was detected
  @param r feature row
  @param c feature col
  @param curv_thr high threshold on ratio of principal curvatures

  @return Returns 0 if the feature at (r,c) in dog_img is sufficiently
  corner-like or 1 otherwise.
  */
static int sift_runtime::is_too_edge_like(mv_image_t* dog_img, int r, int c, int curv_thr)
{
    double d, dxx, dyy, dxy, tr, det;

    /* principal curvatures are computed using the trace and det of Hessian */
    d = pixval32f(dog_img, r, c);
    dxx = pixval32f(dog_img, r, c + 1) + pixval32f(dog_img, r, c - 1) - 2 * d;
    dyy = pixval32f(dog_img, r + 1, c) + pixval32f(dog_img, r - 1, c) - 2 * d;
    dxy = (pixval32f(dog_img, r + 1, c + 1) - pixval32f(dog_img, r + 1, c - 1) -
        pixval32f(dog_img, r - 1, c + 1) + pixval32f(dog_img, r - 1, c - 1)) / 4.0;
    tr = dxx + dyy;
    det = dxx * dyy - dxy * dxy;

    /* negative determinant -> curvatures have different signs; reject feature */
    if (det <= 0)
        return 1;

    if (tr * tr / det < (curv_thr + 1.0)*(curv_thr + 1.0) / curv_thr)
        return 0;
    return 1;
}




/*
  Calculates characteristic scale for each feature in an array.

  @param features array of features
  @param sigma amount of Gaussian smoothing per octave of scale space
  @param intvls intervals per octave of scale space
  */
static void sift_runtime::calc_feature_scales(mv_features* features, double sigma, int intvls)
{
    struct feature* feat;
    struct detection_data* ddata;
    double intvl;

    int n = features->size();
    for (int i = 0; i < n; i++)
    {
        feat = features->at(i);
        ddata = feat_detection_data(feat);
        intvl = ddata->intvl + ddata->subintvl;
        feat->scl = sigma * pow(2.0, ddata->octv + intvl / intvls);
        ddata->scl_octv = sigma * pow(2.0, intvl / intvls);
    }
}



/*
  Halves feature coordinates and scale in case the input image was doubled
  prior to scale space construction.

  @param features array of features
  */
static void sift_runtime::adjust_for_img_dbl(mv_features* features)
{
    struct feature* feat;
    int i;

    int n = features->size();
    for (int i = 0; i < n; i++)
    {
        feat = features->at(i);
        feat->x /= 2.0;
        feat->y /= 2.0;
        feat->scl /= 2.0;
        feat->img_pt.x /= 2.0;
        feat->img_pt.y /= 2.0;
    }
}



/*
  Computes a canonical orientation for each image feature in an array.  Based
  on Section 5 of Lowe's paper.  This function adds features to the array when
  there is more than one dominant orientation at a given feature location.

  @param features an array of image features
  @param gauss_pyr Gaussian scale space pyramid
  */
static void sift_runtime::calc_feature_oris(mv_features* features, mv_image_t*** gauss_pyr)
{
    struct feature* feat;
    struct detection_data* ddata;
    double* hist;
    double omax;
    int i, j;

    int n = features->size();

    for (int i = 0; i < n; i++)
    {        
        //feat = features->at(i);
        //feat = (struct feature*)malloc(sizeof(struct feature));
        feat = features->back();
        features->pop_back();
        ddata = feat_detection_data(feat);

        hist = ori_hist(gauss_pyr[ddata->octv][ddata->intvl], ddata->r, ddata->c, SIFT_ORI_HIST_BINS,
            mv_round(SIFT_ORI_RADIUS * ddata->scl_octv), SIFT_ORI_SIG_FCTR * ddata->scl_octv);

        for (int j = 0; j < SIFT_ORI_SMOOTH_PASSES; j++)
            smooth_ori_hist(hist, SIFT_ORI_HIST_BINS);

        omax = dominant_ori(hist, SIFT_ORI_HIST_BINS);
        add_good_ori_features(features, hist, SIFT_ORI_HIST_BINS, omax * SIFT_ORI_PEAK_RATIO, feat);

        free(ddata);
        free(feat);
        free(hist);

    }
}



/*
  Computes a gradient orientation histogram at a specified pixel.

  @param img image
  @param r pixel row
  @param c pixel col
  @param n number of histogram bins
  @param rad radius of region over which histogram is computed
  @param sigma std for Gaussian weighting of histogram entries

  @return Returns an n-element array containing an orientation histogram
  representing orientations between 0 and 2 PI.
  */
static double* sift_runtime::ori_hist(mv_image_t* img, int r, int c, int n, int rad,
    double sigma)
{
    double* hist;
    double mag, ori, w, exp_denom, PI2 = MV_PI * 2.0;
    int bin, i, j;

    hist = (double*)calloc(n, sizeof(double));
    exp_denom = 2.0 * sigma * sigma;
    for (i = -rad; i <= rad; i++)
    for (j = -rad; j <= rad; j++)
    if (calc_grad_mag_ori(img, r + i, c + j, &mag, &ori))
    {
        w = exp(-(i*i + j*j) / exp_denom);
        bin = mv_round(n * (ori + MV_PI) / PI2);
        bin = (bin < n) ? bin : 0;
        hist[bin] += w * mag;
    }

    return hist;
}



/*
  Calculates the gradient magnitude and orientation at a given pixel.

  @param img image
  @param r pixel row
  @param c pixel col
  @param mag output as gradient magnitude at pixel (r,c)
  @param ori output as gradient orientation at pixel (r,c)

  @return Returns 1 if the specified pixel is a valid one and sets mag and
  ori accordingly; otherwise returns 0
  */
static int sift_runtime::calc_grad_mag_ori(mv_image_t* img, int r, int c, double* mag,
    double* ori)
{
    double dx, dy;

    if (r > 0 && r < img->height - 1 && c > 0 && c < img->width - 1)
    {
        dx = pixval32f(img, r, c + 1) - pixval32f(img, r, c - 1);
        dy = pixval32f(img, r - 1, c) - pixval32f(img, r + 1, c);
        *mag = sqrt(dx*dx + dy*dy);
        *ori = atan2(dy, dx);
        return 1;
    }

    else
        return 0;
}



/*
  Gaussian smooths an orientation histogram.

  @param hist an orientation histogram
  @param n number of bins
  */
static void sift_runtime::smooth_ori_hist(double* hist, int n)
{
    double prev, tmp, h0 = hist[0];
    int i;

    prev = hist[n - 1];
    for (i = 0; i < n; i++)
    {
        tmp = hist[i];
        hist[i] = 0.25 * prev + 0.5 * hist[i] +
            0.25 * ((i + 1 == n) ? h0 : hist[i + 1]);
        prev = tmp;
    }
}



/*
  Finds the magnitude of the dominant orientation in a histogram

  @param hist an orientation histogram
  @param n number of bins

  @return Returns the value of the largest bin in hist
  */
static double sift_runtime::dominant_ori(double* hist, int n)
{
    double omax;
    int maxbin, i;

    omax = hist[0];
    maxbin = 0;
    for (i = 1; i < n; i++)
    if (hist[i] > omax)
    {
        omax = hist[i];
        maxbin = i;
    }
    return omax;
}



/*
  Interpolates a histogram peak from left, center, and right values
  */
#define interp_hist_peak( l, c, r ) ( 0.5 * ((l)-(r)) / ((l) - 2.0*(c) + (r)) )



/*
  Adds features to an array for every orientation in a histogram greater than
  a specified threshold.

  @param features new features are added to the end of this array
  @param hist orientation histogram
  @param n number of bins in hist
  @param mag_thr new features are added for entries in hist greater than this
  @param feat new features are clones of this with different orientations
  */
static void sift_runtime::add_good_ori_features(mv_features* features, double* hist, int n,
    double mag_thr, struct feature* feat)
{
    struct feature* new_feat;
    double bin, PI2 = MV_PI * 2.0;
    int l, r, i;

    for (i = 0; i < n; i++)
    {
        l = (i == 0) ? n - 1 : i - 1;
        r = (i + 1) % n;

        if (hist[i] > hist[l] && hist[i] > hist[r] && hist[i] >= mag_thr)
        {
            bin = i + interp_hist_peak(hist[l], hist[i], hist[r]);
            bin = (bin < 0) ? n + bin : (bin >= n) ? bin - n : bin;
            new_feat = clone_feature(feat);
            new_feat->ori = ((PI2 * bin) / n) - MV_PI;
            features->push_back(new_feat);
            //free(new_feat);
        }
    }
}



/*
  Makes a deep copy of a feature

  @param feat feature to be cloned

  @return Returns a deep copy of feat
  */
static struct feature* sift_runtime::clone_feature(struct feature* feat)
{
    struct feature* new_feat;
    struct detection_data* ddata;

    new_feat = new_feature();
    ddata = feat_detection_data(new_feat);
    memcpy(new_feat, feat, sizeof(struct feature));
    memcpy(ddata, feat_detection_data(feat), sizeof(struct detection_data));
    new_feat->feature_data = ddata;

    return new_feat;
}



/*
  Computes feature descriptors for features in an array.  Based on Section 6
  of Lowe's paper.

  @param features array of features
  @param gauss_pyr Gaussian scale space pyramid
  @param d width of 2D array of orientation histograms
  @param n number of bins per orientation histogram
  */
static void sift_runtime::compute_descriptors(mv_features* features, mv_image_t*** gauss_pyr, int d, int n)
{
    struct feature* feat;
    struct detection_data* ddata;
    double*** hist;


    int k = features->size();
    for (int i = 0; i < k; i++)
    {
        feat = features->at(i);
        ddata = feat_detection_data(feat);
        hist = descr_hist(gauss_pyr[ddata->octv][ddata->intvl], ddata->r,
            ddata->c, feat->ori, ddata->scl_octv, d, n);
        hist_to_descr(hist, d, n, feat);
        release_descr_hist(&hist, d);
    }
}



/*
  Computes the 2D array of orientation histograms that form the feature
  descriptor.  Based on Section 6.1 of Lowe's paper.

  @param img image used in descriptor computation
  @param r row coord of center of orientation histogram array
  @param c column coord of center of orientation histogram array
  @param ori canonical orientation of feature whose descr is being computed
  @param scl scale relative to img of feature whose descr is being computed
  @param d width of 2d array of orientation histograms
  @param n bins per orientation histogram

  @return Returns a d x d array of n-bin orientation histograms.
  */
static double*** sift_runtime::descr_hist(mv_image_t* img, int r, int c, double ori,
    double scl, int d, int n)
{
    double*** hist;
    double cos_t, sin_t, hist_width, exp_denom, r_rot, c_rot, grad_mag,
        grad_ori, w, rbin, cbin, obin, bins_per_rad, PI2 = 2.0 * MV_PI;
    int radius, i, j;

    hist = (double***)calloc(d, sizeof(double**));
    for (i = 0; i < d; i++)
    {
        hist[i] = (double**)calloc(d, sizeof(double*));
        for (j = 0; j < d; j++)
            hist[i][j] = (double*)calloc(n, sizeof(double));
    }

    cos_t = cos(ori);
    sin_t = sin(ori);
    bins_per_rad = n / PI2;
    exp_denom = d * d * 0.5;
    hist_width = SIFT_DESCR_SCL_FCTR * scl;
    radius = hist_width * sqrt(2.0) * (d + 1.0) * 0.5 + 0.5;
    for (i = -radius; i <= radius; i++)
    for (j = -radius; j <= radius; j++)
    {
        /*
          Calculate sample's histogram array coords rotated relative to ori.
          Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
          r_rot = 1.5) have full weight placed in row 1 after interpolation.
          */
        c_rot = (j * cos_t - i * sin_t) / hist_width;
        r_rot = (j * sin_t + i * cos_t) / hist_width;
        rbin = r_rot + d / 2 - 0.5;
        cbin = c_rot + d / 2 - 0.5;

        if (rbin > -1.0  &&  rbin < d  &&  cbin > -1.0  &&  cbin < d)
        if (calc_grad_mag_ori(img, r + i, c + j, &grad_mag, &grad_ori))
        {
            grad_ori -= ori;
            while (grad_ori < 0.0)
                grad_ori += PI2;
            while (grad_ori >= PI2)
                grad_ori -= PI2;

            obin = grad_ori * bins_per_rad;
            w = exp(-(c_rot * c_rot + r_rot * r_rot) / exp_denom);
            interp_hist_entry(hist, rbin, cbin, obin, grad_mag * w, d, n);
        }
    }

    return hist;
}



/*
  Interpolates an entry into the array of orientation histograms that form
  the feature descriptor.

  @param hist 2D array of orientation histograms
  @param rbin sub-bin row coordinate of entry
  @param cbin sub-bin column coordinate of entry
  @param obin sub-bin orientation coordinate of entry
  @param mag size of entry
  @param d width of 2D array of orientation histograms
  @param n number of bins per orientation histogram
  */
static void interp_hist_entry(double*** hist, double rbin, double cbin,
    double obin, double mag, int d, int n)
{
    double d_r, d_c, d_o, v_r, v_c, v_o;
    double** row, *h;
    int r0, c0, o0, rb, cb, ob, r, c, o;

    r0 = mv_floor(rbin);
    c0 = mv_floor(cbin);
    o0 = mv_floor(obin);
    d_r = rbin - r0;
    d_c = cbin - c0;
    d_o = obin - o0;

    /*
      The entry is distributed into up to 8 bins.  Each entry into a bin
      is multiplied by a weight of 1 - d for each dimension, where d is the
      distance from the center value of the bin measured in bin units.
      */
    for (r = 0; r <= 1; r++)
    {
        rb = r0 + r;
        if (rb >= 0 && rb < d)
        {
            v_r = mag * ((r == 0) ? 1.0 - d_r : d_r);
            row = hist[rb];
            for (c = 0; c <= 1; c++)
            {
                cb = c0 + c;
                if (cb >= 0 && cb < d)
                {
                    v_c = v_r * ((c == 0) ? 1.0 - d_c : d_c);
                    h = row[cb];
                    for (o = 0; o <= 1; o++)
                    {
                        ob = (o0 + o) % n;
                        v_o = v_c * ((o == 0) ? 1.0 - d_o : d_o);
                        h[ob] += v_o;
                    }
                }
            }
        }
    }
}



/*
  Converts the 2D array of orientation histograms into a feature's descriptor
  vector.

  @param hist 2D array of orientation histograms
  @param d width of hist
  @param n bins per histogram
  @param feat feature into which to store descriptor
  */
static void hist_to_descr(double*** hist, int d, int n, struct feature* feat)
{
    int int_val, i, r, c, o, k = 0;

    for (r = 0; r < d; r++)
    for (c = 0; c < d; c++)
    for (o = 0; o < n; o++)
        feat->descr[k++] = hist[r][c][o];

    feat->d = k;
    normalize_descr(feat);
    for (i = 0; i < k; i++)
    if (feat->descr[i] > SIFT_DESCR_MAG_THR)
        feat->descr[i] = SIFT_DESCR_MAG_THR;
    normalize_descr(feat);

    /* convert floating-point descriptor to integer valued descriptor */
    for (i = 0; i < k; i++)
    {
        int_val = SIFT_INT_DESCR_FCTR * feat->descr[i];
        feat->descr[i] = MIN(255, int_val);
    }
}


/*
  Normalizes a feature's descriptor vector to unitl length

  @param feat feature
  */
static void normalize_descr(struct feature* feat)
{
    double cur, len_inv, len_sq = 0.0;
    int i, d = feat->d;

    for (i = 0; i < d; i++)
    {
        cur = feat->descr[i];
        len_sq += cur*cur;
    }
    len_inv = 1.0 / sqrt(len_sq);
    for (i = 0; i < d; i++)
        feat->descr[i] *= len_inv;
}



/*
  Compares features for a decreasing-scale ordering.  Intended for use with
  CvSeqSort

  @param feat1 first feature
  @param feat2 second feature
  @param param unused

  @return Returns 1 if feat1's scale is greater than feat2's, -1 if vice versa,
  and 0 if their scales are equal
  */
static int feature_cmp(void* feat1, void* feat2, void* param)
{
    struct feature* f1 = (struct feature*) feat1;
    struct feature* f2 = (struct feature*) feat2;

    if (f1->scl < f2->scl)
        return 1;
    if (f1->scl > f2->scl)
        return -1;
    return 0;
}



/*
  De-allocates memory held by a descriptor histogram

  @param hist pointer to a 2D array of orientation histograms
  @param d width of hist
  */
static void release_descr_hist(double**** hist, int d)
{
    int i, j;

    for (i = 0; i < d; i++)
    {
        for (j = 0; j < d; j++)
            free((*hist)[i][j]);
        free((*hist)[i]);
    }
    free(*hist);
    *hist = NULL;
}


/*
  De-allocates memory held by a scale space pyramid

  @param pyr scale space pyramid
  @param octvs number of octaves of scale space
  @param n number of images per octave
  */
static void release_pyr(mv_image_t**** pyr, int octvs, int n)
{
    int i, j;
    for (i = 0; i < octvs; i++)
    {
        for (j = 0; j < n; j++)
            mv_release_image(&(*pyr)[i][j]);
        free((*pyr)[i]);
    }
    free(*pyr);
    *pyr = NULL;
}



sift_runtime::sift_runtime() 
{
    m_pool_used = 0;
    m_octvs_step = 0;
    memset(m_gauss_pyr, 0, sizeof(m_gauss_pyr));
    memset(m_dog_pyr, 0, sizeof(m_dog_pyr));
}


sift_runtime::~sift_runtime()
{

}

int sift_runtime::process(mv_image_t* img, mv_features* features)
{    
    assert(img != NULL || features != NULL);

    // build scale space pyramid; smallest dimension of top level is ~4 pixels 
    mv_image_t* init_img = create_init_img(img, SIFT_IMG_DBL, SIFT_SIGMA);

    m_octvs_step = log((double)(MIN(init_img->width, init_img->height))) / log(2.0) - 2;

    build_gauss_pyr(init_img, m_octvs_step, SIFT_INTVLS, SIFT_SIGMA);
    build_dog_pyr(gauss_pyr, m_octvs_step, SIFT_INTVLS);


    int ret = scale_space_extrema(dog_pyr, octvs, SIFT_INTVLS, SIFT_CONTR_THR, SIFT_CURV_THR, features);
    if (ret != 0) return -1;

    WRITE_INFO_LOG("features size %d ", features->size());

    calc_feature_scales(features, SIFT_SIGMA, SIFT_INTVLS);

    if (SIFT_IMG_DBL)
        adjust_for_img_dbl(features);

    calc_feature_oris(features, gauss_pyr);
    compute_descriptors(features, gauss_pyr, SIFT_DESCR_WIDTH, SIFT_DESCR_HIST_BINS);

    /* sort features by decreasing scale and move from CvSeq to array */
    int i, n = 0;

    //mv_seq_sort(features, (CvCmpFunc)feature_cmp, NULL);
    /*n = features->total;
    *feat = (struct feature*)calloc(n, sizeof(struct feature));
    *feat = (struct feature*)mv_cvt_seq_2_array(features, *feat, MV_WHOLE_SEQ);
    for (i = 0; i < n; i++)
    {
    free((*feat)[i].feature_data);
    (*feat)[i].feature_data = NULL;
    }*/

    //mv_release_storage(&storage);
    mv_release_image(&init_img);
    release_pyr(&gauss_pyr, octvs, SIFT_INTVLS + 3);
    release_pyr(&dog_pyr, octvs, SIFT_INTVLS + 2);

    return n;
}



mv_image_t* sift_runtime::create_init_img(mv_image_t* img, int img_dbl, double sigma)
{
    mv_image_t* gray = convert_to_gray32(img);

    if (img_dbl)
    {
        double sig_diff = sqrt(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4);
        mv_image_t* dbl = mv_create_image(mv_size_t(img->width * 2, img->height * 2), IPL_DEPTH_32F, 1);

        //show_temp_image(gray);
        mv_resize_cubic(gray, dbl);
        //show_temp_image(dbl);

        mv_box_blur(dbl, dbl, sig_diff);
        //show_temp_image(dbl);

        mv_release_image(&gray);
        return dbl;
    }
    else
    {
        double sig_diff = sqrt(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA);
        mv_box_blur(gray, gray, sig_diff);
        return gray;
    }
}


mv_image_t* sift_runtime::convert_to_gray32(mv_image_t* img)
{   
    mv_image_t* gray32 = mv_create_image(mv_get_size(img), IPL_DEPTH_32F, 1);

    mv_image_t* gray8;
    if (img->nChannels == 1) {
        gray8 = (mv_image_t*)mv_clone_image(img);

    } else {
        gray8 = mv_create_image(mv_get_size(img), IPL_DEPTH_8U, 1);
        mv_convert_gray(img, gray8);
    }

    mv_normalize_u8(gray8, gray32, 1.0 / 255.0);

    mv_release_image(&gray8);
    return gray32;
}
