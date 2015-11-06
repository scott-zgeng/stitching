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
int sift_runtime::build_gauss_pyramid(mv_image_t* base)
{
    /* 
      precompute Gaussian sigmas using the following formula:

      \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2

      sig[i] is the incremental sigma value needed to compute
      the actual sigma of level i. Keeping track of incremental
      sigmas vs. total sigmas keeps the gaussian kernel small.
      */
    double k = pow(2.0, 1.0 / SIFT_INTVLS);
    double sig[SIFT_INTVLS + 3];

    sig[0] = SIFT_SIGMA;
    sig[1] = SIFT_SIGMA * sqrt(k*k - 1);
    for (int i = 2; i < SIFT_INTVLS + 3; i++) {
        sig[i] = sig[i - 1] * k;
    }

    int octvs = m_octvs;
    for (int o = 0; o < octvs; o++) {
        for (int i = 0; i < SIFT_INTVLS + 3; i++) {
            if (o == 0 && i == 0) {
                set_gauss_pyramid(mv_clone_image(base), o, i);
            }
            else if (i == 0) {
                /* base of new octvave is halved image from end of previous octave */
                set_gauss_pyramid(downsample(gauss_pyramid(o - 1, SIFT_INTVLS)), o, i);
            }            
            else {
                /* blur the current octave's last image to create the next one */
                mv_image_t* prev_layer = gauss_pyramid(o, i - 1);
                set_gauss_pyramid(mv_create_image(mv_get_size(prev_layer), IPL_DEPTH_32F, 1), o, i);
                mv_box_blur(prev_layer, gauss_pyramid(o, i), sig[i]);
            }
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
int sift_runtime::build_dog_pyramid()
{    
    int octvs = m_octvs;
    for (int o = 0; o < octvs; o++) {
        for (int i = 0; i < SIFT_INTVLS + 2; i++) {
            mv_image_t* prev = gauss_pyramid(o, i + 1);
            mv_image_t* curr = gauss_pyramid(o, i);
            mv_image_t* dog = mv_create_image(mv_get_size(curr), IPL_DEPTH_32F, 1);
            set_dog_pyramid(dog, o, i);
            mv_sub(prev, curr, dog);

            WRITE_INFO_LOG("SUB o = %d i = %d", o, i);
            show_temp_image(dog);
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

int sift_runtime::scale_space_extrema()
{
    int octvs = m_octvs;
    
    feature* feat;
    detection_data* ddata;
    pyramid_layer dogs;

    for (int o = 0; o < octvs; o++) {
        int height = dog_pyramid(o, 0)->height;
        int width = dog_pyramid(o, 0)->width;

        for (int i = 1; i <= SIFT_INTVLS; i++) {
            
            dogs.curr = dog_pyramid(o, i);
            dogs.prev = dog_pyramid(o, i - 1);
            dogs.next = dog_pyramid(o, i + 1);
            
            for (int r = SIFT_IMG_BORDER; r < height - SIFT_IMG_BORDER; r++) {
                for (int c = SIFT_IMG_BORDER; c < width - SIFT_IMG_BORDER; c++) {
                    
                    // perform preliminary check on contrast 
                    if (ABS(pixval32f(dogs.curr, r, c)) <= PRELIM_CONTR_THR)
                        continue;

                    if (!is_extremum(&dogs, r, c))
                        continue;

                    feat = interp_extremum(&dogs, o, i, r, c);
                    if (!feat)
                        continue;
                    
                    ddata = &feat->ddata;
                    mv_image_t* temp = dog_pyramid(ddata->octv, ddata->intvl);
                    if (is_too_edge_like(temp, ddata->r, ddata->c)) {
                        WRITE_INFO_LOG("keypoint o=%d, i=%d, r=%d, c=%d", o, i, r, c);
                        release_last_feature();                                    
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
bool sift_runtime::is_extremum(pyramid_layer* dogs, int r, int c)
{
    double val = pixval32f(dogs->curr, r, c);
    
    // check for maximum 
    if (val > 0)
    {
        for (int j = -1; j <= 1; j++)
        for (int k = -1; k <= 1; k++)
        if (val < pixval32f(dogs->prev, r + j, c + k)) return false;

        for (int j = -1; j <= 1; j++)
        for (int k = -1; k <= 1; k++)
        if (val < pixval32f(dogs->curr, r + j, c + k)) return false;

        for (int j = -1; j <= 1; j++)
        for (int k = -1; k <= 1; k++)
        if (val < pixval32f(dogs->next, r + j, c + k)) return false;
    }

    // check for minimum 
    else
    {        
        for (int j = -1; j <= 1; j++)
        for (int k = -1; k <= 1; k++)
        if (val > pixval32f(dogs->prev, r + j, c + k)) return false;

        for (int j = -1; j <= 1; j++)
        for (int k = -1; k <= 1; k++)
        if (val > pixval32f(dogs->curr, r + j, c + k)) return false;

        for (int j = -1; j <= 1; j++)
        for (int k = -1; k <= 1; k++)
        if (val > pixval32f(dogs->next, r + j, c + k)) return false;
    }

    return true;
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


struct feature* sift_runtime::interp_extremum(pyramid_layer* dogs, int octv, int intvl, int r, int c)
{
    struct feature* feat;
    struct detection_data* ddata;
    double xi, xr, xc, contr;
    int i = 0;

    while (i < SIFT_MAX_INTERP_STEPS)
    {
        interp_step(dogs, r, c, &xi, &xr, &xc);
        if (ABS(xi) < 0.5  &&  ABS(xr) < 0.5  &&  ABS(xc) < 0.5)
            break;

        c += mv_round(xc);
        r += mv_round(xr);
        intvl += mv_round(xi);

        if (intvl < 1 || intvl > SIFT_INTVLS || 
            c < SIFT_IMG_BORDER || c >= dogs->curr->width - SIFT_IMG_BORDER || 
            r < SIFT_IMG_BORDER || r >= dogs->curr->height - SIFT_IMG_BORDER)
        {
            return NULL;
        }

        i++;
    }

    /* ensure convergence of interpolation */
    if (i >= SIFT_MAX_INTERP_STEPS)
        return NULL;

    contr = interp_contr(dogs, r, c, xi, xr, xc);
    if (ABS(contr) < SIFT_CONTR_THR / SIFT_INTVLS)
        return NULL;

    feat = new_feature();
    ddata = &feat->ddata;
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

void sift_runtime::interp_step(pyramid_layer* dogs, int r, int c, double* xi, double* xr, double* xc)
{
    double deriv_v[3];
    deriv_3D(dogs, r, c, deriv_v);
    mv_mat_handle dD = mv_create_matrix(3, 1);
    mv_matrix_set(dD, 0, 0, -deriv_v[0]);
    mv_matrix_set(dD, 1, 0, -deriv_v[1]);
    mv_matrix_set(dD, 2, 0, -deriv_v[2]);

    mv_mat_handle H = hessian_3D(dogs, r, c);
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

void sift_runtime::deriv_3D(pyramid_layer* dogs, int r, int c, double* result)
{
    double dx = (pixval32f(dogs->curr, r, c + 1) - pixval32f(dogs->curr, r, c - 1)) / 2.0;
    double dy = (pixval32f(dogs->curr, r + 1, c) - pixval32f(dogs->curr, r - 1, c)) / 2.0;
    double ds = (pixval32f(dogs->next, r, c) - pixval32f(dogs->prev, r, c)) / 2.0;

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
mv_mat_handle sift_runtime::hessian_3D(pyramid_layer* dogs, int r, int c)
{
    double v = pixval32f(dogs->curr, r, c);
    
    double dxx = (pixval32f(dogs->curr, r, c + 1) + pixval32f(dogs->curr, r, c - 1) - 2 * v);
    double dyy = (pixval32f(dogs->curr, r + 1, c) + pixval32f(dogs->curr, r - 1, c) - 2 * v);
    double dss = (pixval32f(dogs->next, r, c) + pixval32f(dogs->prev, r, c) - 2 * v);

    double dxy = (pixval32f(dogs->curr, r + 1, c + 1) - pixval32f(dogs->curr, r + 1, c - 1) -
        pixval32f(dogs->curr, r - 1, c + 1) + pixval32f(dogs->curr, r - 1, c - 1)) / 4.0;
    double dxs = (pixval32f(dogs->next, r, c + 1) - pixval32f(dogs->next, r, c - 1) -
        pixval32f(dogs->prev, r, c + 1) + pixval32f(dogs->prev, r, c - 1)) / 4.0;
    double dys = (pixval32f(dogs->next, r + 1, c) - pixval32f(dogs->next, r - 1, c) -
        pixval32f(dogs->prev, r + 1, c) + pixval32f(dogs->prev, r - 1, c)) / 4.0;

    mv_mat_handle H = mv_create_matrix(3, 3);
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
double sift_runtime::interp_contr(pyramid_layer* dogs, int r, int c, double xi, double xr, double xc)
{
    mv_mat_handle X = mv_create_matrix(3, 1);
    mv_matrix_set(X, 0, 0, xc);
    mv_matrix_set(X, 0, 1, xr);
    mv_matrix_set(X, 0, 2, xi);

    mv_mat_handle T = mv_create_matrix(1, 1);

    double deriv_x[3];
    deriv_3D(dogs, r, c, deriv_x);
    mv_mat_handle dD = mv_create_matrix(1, 3);
    mv_matrix_set(dD, 0, 0, deriv_x[0]);
    mv_matrix_set(dD, 0, 1, deriv_x[1]);
    mv_matrix_set(dD, 0, 2, deriv_x[2]);
    mv_matrix_mul(dD, X, T);

    mv_release_matrix(dD);

    double t = mv_matrix_get(T, 0, 0);
    return pixval32f(dogs->curr, r, c) + t * 0.5;
}



/*
  Allocates and initializes a new feature

  @return Returns a pointer to the new feature
  */
struct feature* sift_runtime::new_feature()
{
    if (m_pool_used == MAX_FEATURE_SIZE)
        return NULL;

    feature* feat = m_pool + m_pool_used;
    m_pool_used++;

    memset(feat, 0, sizeof(struct feature));
    feat->type = FEATURE_LOWE;    
    feat = (struct feature*)malloc(sizeof(struct feature));

    return feat;
}

void sift_runtime::release_last_feature()
{
    if (m_pool_used > 0)
        m_pool_used--;
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
bool sift_runtime::is_too_edge_like(mv_image_t* dog, int r, int c)
{
    // principal curvatures are computed using the trace and det of Hessian 
    double d = pixval32f(dog, r, c);
    double dxx = pixval32f(dog, r, c + 1) + pixval32f(dog, r, c - 1) - 2 * d;
    double dyy = pixval32f(dog, r + 1, c) + pixval32f(dog, r - 1, c) - 2 * d;
    double dxy = (pixval32f(dog, r + 1, c + 1) - pixval32f(dog, r + 1, c - 1) -
        pixval32f(dog, r - 1, c + 1) + pixval32f(dog, r - 1, c - 1)) / 4.0;
    double tr = dxx + dyy;
    double det = dxx * dyy - dxy * dxy;

    // negative determinant -> curvatures have different signs; reject feature 
    if (det <= 0)
        return true;

    if (tr * tr / det < (SIFT_CURV_THR + 1.0)*(SIFT_CURV_THR + 1.0) / SIFT_CURV_THR)
        return false;
    return true;
}




/*
  Calculates characteristic scale for each feature in an array.

  @param features array of features
  @param sigma amount of Gaussian smoothing per octave of scale space
  @param intvls intervals per octave of scale space
  */

void sift_runtime::calc_feature_scales()
{
    struct feature* feat;
    struct detection_data* ddata;
    double intvl;

    int n = m_pool_used;
    for (int i = 0; i < n; i++)
    {
        feat = m_pool + i;
        ddata = &feat->ddata;
        intvl = ddata->intvl + ddata->subintvl;
        feat->scl = SIFT_SIGMA * pow(2.0, ddata->octv + intvl / SIFT_INTVLS);
        ddata->scl_octv = SIFT_SIGMA * pow(2.0, intvl / SIFT_INTVLS);
    }
}



/*
  Halves feature coordinates and scale in case the input image was doubled
  prior to scale space construction.

  @param features array of features
  */
void sift_runtime::adjust_for_img_dbl()
{
    struct feature* feat;

    int n = m_pool_used;
    for (int i = 0; i < n; i++)
    {
        feat = m_pool + i;
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
void sift_runtime::calc_feature_oris()
{
    struct feature* feat;
    struct detection_data* ddata;
    double hist[SIFT_ORI_HIST_BINS];

    int n = m_pool_used;
    for (int i = 0; i < n; i++)
    {
        feat = m_pool + i;
        ddata = &feat->ddata;

        mv_image_t* gauss = gauss_pyramid(ddata->octv, ddata->intvl);
        ori_hist(hist, gauss, ddata->r, ddata->c, mv_round(SIFT_ORI_RADIUS * ddata->scl_octv), SIFT_ORI_SIG_FCTR * ddata->scl_octv);

        for (int j = 0; j < SIFT_ORI_SMOOTH_PASSES; j++)
            smooth_ori_hist(hist);
        
        double omax = dominant_ori(hist);
        adjust_good_ori_features(hist, omax * SIFT_ORI_PEAK_RATIO, feat);
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
void sift_runtime::ori_hist(double* hist, mv_image_t* img, int r, int c, int rad, double sigma)
{    
    double mag, ori;    
    double exp_denom = 2.0 * sigma * sigma;

    for (int i = -rad; i <= rad; i++)
    for (int j = -rad; j <= rad; j++)
    if (calc_grad_mag_ori(img, r + i, c + j, &mag, &ori))
    {
        double w = exp(-(i*i + j*j) / exp_denom);
        int bin = mv_round(SIFT_ORI_HIST_BINS * (ori + MV_PI) / MV_PI2);
        bin = (bin < SIFT_ORI_HIST_BINS) ? bin : 0;
        hist[bin] += w * mag;
    }
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
int sift_runtime::calc_grad_mag_ori(mv_image_t* img, int r, int c, double* mag, double* ori)
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
void sift_runtime::smooth_ori_hist(double* hist)
{
    double tmp, h0 = hist[0];

    double prev = hist[SIFT_ORI_HIST_BINS - 1];

    for (int i = 0; i < SIFT_ORI_HIST_BINS; i++)
    {
        tmp = hist[i];
        hist[i] = 0.25 * prev + 0.5 * hist[i] + 0.25 * ((i + 1 == SIFT_ORI_HIST_BINS) ? h0 : hist[i + 1]);
        prev = tmp;
    }
}



/*
  Finds the magnitude of the dominant orientation in a histogram

  @param hist an orientation histogram
  @param n number of bins

  @return Returns the value of the largest bin in hist
  */
double sift_runtime::dominant_ori(double* hist)
{
    double omax;
    int maxbin, i;

    omax = hist[0];
    maxbin = 0;
    for (i = 1; i < SIFT_ORI_HIST_BINS; i++)
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
void sift_runtime::adjust_good_ori_features(double* hist, double mag_thr, feature* feat)
{    
    for (int i = 0; i < SIFT_ORI_HIST_BINS; i++)
    {
        int l = (i == 0) ? SIFT_ORI_HIST_BINS - 1 : i - 1;
        int r = (i + 1) % SIFT_ORI_HIST_BINS;

        if (hist[i] > hist[l] && hist[i] > hist[r] && hist[i] >= mag_thr)
        {
            double bin = i + interp_hist_peak(hist[l], hist[i], hist[r]);
            bin = (bin < 0) ? SIFT_ORI_HIST_BINS + bin : (bin >= SIFT_ORI_HIST_BINS) ? bin - SIFT_ORI_HIST_BINS : bin;
            //new_feat = clone_feature(feat);
            feat->ori = ((MV_PI2 * bin) / SIFT_ORI_HIST_BINS) - MV_PI;
            //features->push_back(new_feat);
            //free(new_feat);
        }
    }
}



/*
  Makes a deep copy of a feature

  @param feat feature to be cloned

  @return Returns a deep copy of feat
  */
//struct feature* sift_runtime::clone_feature(struct feature* feat)
//{
//    struct feature* new_feat;
//    struct detection_data* ddata;
//
//    new_feat = new_feature();
//    ddata = feat_detection_data(new_feat);
//    memcpy(new_feat, feat, sizeof(struct feature));
//    memcpy(ddata, feat_detection_data(feat), sizeof(struct detection_data));
//    new_feat->feature_data = ddata;
//
//    return new_feat;
//}



/*
  Computes feature descriptors for features in an array.  Based on Section 6
  of Lowe's paper.

  @param features array of features
  @param gauss_pyr Gaussian scale space pyramid
  @param d width of 2D array of orientation histograms
  @param n number of bins per orientation histogram
  */

void sift_runtime::compute_descriptors()
{
    struct feature* feat;
    struct detection_data* ddata;
    double hist[SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS];

    int k = m_pool_used;
    for (int i = 0; i < k; i++)
    {
        feat = m_pool + i;
        ddata = &feat->ddata;

        mv_image_t* gauss = gauss_pyramid(ddata->octv, ddata->intvl);
        descr_hist(hist, gauss, ddata->r, ddata->c, feat->ori, ddata->scl_octv);
        hist_to_descr(hist, feat);
        
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


void sift_runtime::descr_hist(double* hist, mv_image_t* img, int r, int c, double ori, double scl)
{        

    double grad_mag;
    double grad_ori;

    double cos_t = cos(ori);
    double sin_t = sin(ori);
    double bins_per_rad = SIFT_DESCR_HIST_BINS / MV_PI2;
    double exp_denom = SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * 0.5;
    double hist_width = SIFT_DESCR_SCL_FCTR * scl;
    int radius = hist_width * sqrt(2.0) * (SIFT_DESCR_WIDTH + 1.0) * 0.5 + 0.5;
    for (int i = -radius; i <= radius; i++)
    for (int j = -radius; j <= radius; j++)
    {
        /*
          Calculate sample's histogram array coords rotated relative to ori.
          Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
          r_rot = 1.5) have full weight placed in row 1 after interpolation.
          */
        double c_rot = (j * cos_t - i * sin_t) / hist_width;
        double r_rot = (j * sin_t + i * cos_t) / hist_width;
        double rbin = r_rot + SIFT_DESCR_WIDTH / 2 - 0.5;
        double cbin = c_rot + SIFT_DESCR_WIDTH / 2 - 0.5;

        if (rbin > -1.0  &&  rbin < SIFT_DESCR_WIDTH  &&  cbin > -1.0  &&  cbin < SIFT_DESCR_WIDTH)
        if (calc_grad_mag_ori(img, r + i, c + j, &grad_mag, &grad_ori))
        {
            grad_ori -= ori;
            while (grad_ori < 0.0)
                grad_ori += MV_PI2;
            while (grad_ori >= MV_PI2)
                grad_ori -= MV_PI2;

            double obin = grad_ori * bins_per_rad;
            double w = exp(-(c_rot * c_rot + r_rot * r_rot) / exp_denom);
            interp_hist_entry(hist, rbin, cbin, obin, grad_mag * w);
        }
    }
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
void sift_runtime::interp_hist_entry(double* hist, double rbin, double cbin, double obin, double mag)
{
    double* pos;

    int r0 = mv_floor(rbin);
    int c0 = mv_floor(cbin);
    int o0 = mv_floor(obin);
    double d_r = rbin - r0;
    double d_c = cbin - c0;
    double d_o = obin - o0;

    /*
      The entry is distributed into up to 8 bins.  Each entry into a bin
      is multiplied by a weight of 1 - d for each dimension, where d is the
      distance from the center value of the bin measured in bin units.
      */
    for (int r = 0; r <= 1; r++)
    {
        int rb = r0 + r;
        if (rb >= 0 && rb < SIFT_DESCR_WIDTH)
        {
            double v_r = mag * ((r == 0) ? 1.0 - d_r : d_r);
            //row = hist[rb];
            for (int c = 0; c <= 1; c++)
            {
                int cb = c0 + c;
                if (cb >= 0 && cb < SIFT_DESCR_WIDTH)
                {
                    double v_c = v_r * ((c == 0) ? 1.0 - d_c : d_c);
                    //h = row[cb];
                    for (int o = 0; o <= 1; o++)
                    {
                        int ob = (o0 + o) % SIFT_DESCR_HIST_BINS;
                        double v_o = v_c * ((o == 0) ? 1.0 - d_o : d_o);
                        pos = hist + SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS * rb + SIFT_DESCR_HIST_BINS * cb + ob;
                        *pos += v_o;
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
//SIFT_DESCR_WIDTH, SIFT_DESCR_HIST_BINS
void sift_runtime::hist_to_descr(double* hist, struct feature* feat)
{
    //for (k = 0; k < SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS; k++)
    /*for (c = 0; c < SIFT_DESCR_WIDTH; c++)
    for (o = 0; o < SIFT_DESCR_HIST_BINS; o++)*/

    memcpy(feat->descr, hist, FEATURE_MAX_D * sizeof(double));
    int k = FEATURE_MAX_D;

    feat->d = k;
    normalize_descr(feat);
    for (int i = 0; i < k; i++)
    if (feat->descr[i] > SIFT_DESCR_MAG_THR)
        feat->descr[i] = SIFT_DESCR_MAG_THR;
    normalize_descr(feat);
        
    // convert floating-point descriptor to integer valued descriptor 
    int int_val;
    for (int i = 0; i < k; i++)
    {
        int_val = SIFT_INT_DESCR_FCTR * feat->descr[i];
        feat->descr[i] = MIN(255, int_val);
    }
}


/*
  Normalizes a feature's descriptor vector to unitl length

  @param feat feature
  */
void sift_runtime::normalize_descr(struct feature* feat)
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
int sift_runtime::feature_cmp(const void* feat1, const void* feat2)
{
    struct feature* f1 = (struct feature*) feat1;
    struct feature* f2 = (struct feature*) feat2;

    if (f1->scl < f2->scl)
        return 1;
    if (f1->scl > f2->scl)
        return -1;
    return 0;
}


//
///*
//  De-allocates memory held by a descriptor histogram
//
//  @param hist pointer to a 2D array of orientation histograms
//  @param d width of hist
//  */
//void sift_runtime::release_descr_hist(double**** hist, int d)
//{
//    int i, j;
//
//    for (i = 0; i < d; i++)
//    {
//        for (j = 0; j < d; j++)
//            free((*hist)[i][j]);
//        free((*hist)[i]);
//    }
//    free(*hist);
//    *hist = NULL;
//}


/*
  De-allocates memory held by a scale space pyramid

  @param pyr scale space pyramid
  @param octvs number of octaves of scale space
  @param n number of images per octave
  */
void sift_runtime::release_pyramid()
{    
    int n = MAX_OCTVS * SIFT_INTVLS + MAX_OCTVS * 3;
    for (int i = 0; i < n; i++)  {
        if (m_gauss_pyr[i] != NULL) {
            mv_release_image(&m_gauss_pyr[i]);
        }
    }    

    for (int i = 0; i < n; i++)  {
        if (m_dog_pyr[i] != NULL) {
            mv_release_image(&m_dog_pyr[i]);
        }
    }
}



sift_runtime::sift_runtime() 
{
    m_pool_used = 0;
    m_octvs = 0;
    memset(m_gauss_pyr, 0, sizeof(m_gauss_pyr));
    memset(m_dog_pyr, 0, sizeof(m_dog_pyr));
}


sift_runtime::~sift_runtime()
{

}

void sift_runtime::export_features(mv_features* features)
{
    feature* items[MAX_FEATURE_SIZE];
    for (int i = 0; i < m_pool_used; i++) {
        items[i] = m_pool + i;
    }
    qsort(items, m_pool_used, sizeof(feature*), feature_cmp);

    for (int i = 0; i < m_pool_used; i++) {
        features->push_back(items[i]);
    }
}



int sift_runtime::process(mv_image_t* img, mv_features* features)
{    
    assert(img != NULL || features != NULL);
   
    // build scale space pyramid; smallest dimension of top level is ~4 pixels 
    mv_image_t* base = create_init_img(img, SIFT_IMG_DBL);    
    m_octvs = log((double)(MIN(base->width, base->height))) / log(2.0) - 2;
    if (m_octvs > MAX_OCTVS)
        return -1;

    build_gauss_pyramid(base);
    build_dog_pyramid();

    if (scale_space_extrema() != 0)
        return -1;    

    calc_feature_scales();

    if (SIFT_IMG_DBL)
        adjust_for_img_dbl();

    calc_feature_oris();
    compute_descriptors();

    export_features(features);
    
    mv_release_image(&base);
    release_pyramid();    

    return 0;
}



mv_image_t* sift_runtime::create_init_img(mv_image_t* img, bool is_double)
{
    mv_image_t* gray = mv_create_image(mv_get_size(img), IPL_DEPTH_32F, 1);
    mv_image_t* gray_temp;
    
    if (img->nChannels == 1) {
        gray_temp = (mv_image_t*)mv_clone_image(img);
    } else {
        gray_temp = mv_create_image(mv_get_size(img), IPL_DEPTH_8U, 1);
        mv_convert_gray(img, gray_temp);
    }

    mv_normalize_u8(gray_temp, gray, 1.0 / 255.0);
    mv_release_image(&gray_temp);

    
    if (is_double)
    {
        double sig_diff = sqrt(SIFT_SIGMA * SIFT_SIGMA - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4);
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
        double sig_diff = sqrt(SIFT_SIGMA * SIFT_SIGMA - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA);
        mv_box_blur(gray, gray, sig_diff);
        return gray;
    }
}

