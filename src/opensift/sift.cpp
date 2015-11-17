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

#include <Eigen/Dense>
using namespace Eigen;


#ifdef USE_CV_DEBUG
#include "cv.h"
static void show_temp_image(mv_image_t* img)
{
    const char* IMG_TEST_1 = "test_windows_1";
    IplImage* clone = mv_image_mv2cv(img);
    cvShowImage(IMG_TEST_1, clone);
    cvWaitKey(0);
    cvReleaseImage(&clone);
}
#else 
static void show_temp_image(mv_image_t* img) {}
#endif



class sift_runtime : public sift_module
{
private:
    struct pyramid_layer
    {
        mv_image_t* curr;
        mv_image_t* prev;
        mv_image_t* next;
    };

    // 最多保存的特征数    

    // default number of sampled intervals per octave 
    static const int SIFT_INTVLS = 3;

    // default sigma for initial gaussian smoothing 
    const double SIFT_SIGMA = 1.6;

    // default threshold on keypoint contrast |D(x)| 
    const double SIFT_CONTR_THR = 0.04;

    // default threshold on keypoint ratio of principle curvatures 
    static const int SIFT_CURV_THR = 10;

    // double image size before pyramid construction?
    static const int SIFT_IMG_DBL = 1;

    // default width of descriptor histogram array
    static const int SIFT_DESCR_WIDTH = 4;

    // default number of bins per histogram in descriptor array 
    static const int SIFT_DESCR_HIST_BINS = 8;

    // assumed gaussian blur for input image 
    const double SIFT_INIT_SIGMA = 0.5;

    // width of border in which to ignore keypoints 
    static const int SIFT_IMG_BORDER = 5;

    // maximum steps of keypoint interpolation before failure
    static const int SIFT_MAX_INTERP_STEPS = 5;

    // default number of bins in histogram for orientation assignment 
    static const int SIFT_ORI_HIST_BINS = 36;

    // determines gaussian sigma for orientation assignment 
    const double SIFT_ORI_SIG_FCTR = 1.5;

    // determines the radius of the region used in orientation assignment 
    const double SIFT_ORI_RADIUS = 3.0 * SIFT_ORI_SIG_FCTR;

    // number of passes of orientation histogram smoothing 
    static const int SIFT_ORI_SMOOTH_PASSES = 2;

    // orientation magnitude relative to max that results in new feature 
    const double SIFT_ORI_PEAK_RATIO = 0.8;

    // determines the size of a single descriptor orientation histogram 
    const double SIFT_DESCR_SCL_FCTR = 3.0;

    //  threshold on magnitude of elements of descriptor vector 
    const double SIFT_DESCR_MAG_THR = 0.2;

    // factor used to convert floating-point descriptor to unsigned char 
    const double SIFT_INT_DESCR_FCTR = 512.0;

    static const int MAX_OCTVS = 10;

    const double PRELIM_CONTR_THR = 0.5 * SIFT_CONTR_THR / SIFT_INTVLS;


private:
    feature m_pool[MAX_FEATURE_SIZE];
    int m_pool_used;
    int m_octvs;
    mv_image_t* m_gauss_pyr[MAX_OCTVS][SIFT_INTVLS + 3];
    mv_image_t* m_dog_pyr[MAX_OCTVS][SIFT_INTVLS + 3];

public:
    sift_runtime() {
        m_pool_used = 0;
        m_octvs = 0;
        memset(m_gauss_pyr, 0, sizeof(m_gauss_pyr));
        memset(m_dog_pyr, 0, sizeof(m_dog_pyr));
    }

    virtual ~sift_runtime() {
    }


    // Finda SIFT features in an image using user-specified parameter values. 
    //   All detected features are stored in the array pointed to by \a feat.
    virtual int process(mv_image_t* img, mv_features* features) {
        assert(img != NULL || features != NULL);

        // build scale space pyramid; smallest dimension of top level is ~4 pixels 
        mv_image_t* base = create_init_img(img, SIFT_IMG_DBL);
        m_octvs = mv_floor(log((double)(MIN(base->width, base->height))) / log(2.0)) - 2;
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


private:

    inline mv_image_t*& gauss_pyramid(int octvs, int intval) {
        return m_gauss_pyr[octvs][intval];
    }

  
    inline mv_image_t*& dog_pyramid(int octvs, int intval) {
        return m_dog_pyr[octvs][intval];
    }


    mv_image_t* create_init_img(mv_image_t* img, bool is_double) {

        mv_image_t* gray = mv_create_image(mv_get_size(img), IPL_DEPTH_32F, 1);
        mv_image_t* gray_temp;

        if (img->nChannels == 1) {
            gray_temp = (mv_image_t*)mv_clone_image(img);
        }
        else {
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


    // Builds Gaussian scale space pyramid from an image    
    int build_gauss_pyramid(mv_image_t* base) {
        
        // precompute Gaussian sigmas using the following formula:
        // {total}^2 = {i}^2 + {i-1}^2
        // sig[i] is the incremental sigma value needed to compute the actual sigma of level i. 
        // Keeping track of incremental sigmas vs. total sigmas keeps the gaussian kernel small.        
        
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
                    gauss_pyramid(o, i) = mv_clone_image(base);
                    //gauss_pyramid(mv_clone_image(base), o, i);
                }
                else if (i == 0) {
                    /* base of new octvave is halved image from end of previous octave */
                    gauss_pyramid(o, i) = downsample(gauss_pyramid(o - 1, SIFT_INTVLS));
                    //set_gauss_pyramid(downsample(gauss_pyramid(o - 1, SIFT_INTVLS)), o, i);
                }
                else {
                    /* blur the current octave's last image to create the next one */
                    mv_image_t* prev_layer = gauss_pyramid(o, i - 1);
                    gauss_pyramid(o, i) = mv_create_image(mv_get_size(prev_layer), IPL_DEPTH_32F, 1);
                    //set_gauss_pyramid(mv_create_image(mv_get_size(prev_layer), IPL_DEPTH_32F, 1), o, i);
                    mv_box_blur(prev_layer, gauss_pyramid(o, i), sig[i]);
                }
            }
        }

        return 0;
    }

    
    // Builds a difference of Gaussians scale space pyramid by subtracting adjacent
    // intervals of a Gaussian pyramid
    int build_dog_pyramid()
    {
        int octvs = m_octvs;
        for (int o = 0; o < octvs; o++) {
            for (int i = 0; i < SIFT_INTVLS + 2; i++) {
                mv_image_t* prev = gauss_pyramid(o, i + 1);
                mv_image_t* curr = gauss_pyramid(o, i);
                mv_image_t* dog = mv_create_image(mv_get_size(curr), IPL_DEPTH_32F, 1);
                dog_pyramid(o, i) = dog;
                mv_sub(prev, curr, dog);

                WRITE_INFO_LOG("SUB o = %d i = %d", o, i);
                //show_temp_image(dog);
            }
        }

        return 0;
    }


    
    // Downsamples an image to a quarter of its size (half in each dimension)
    // using nearest-neighbor interpolation    
    mv_image_t* downsample(mv_image_t* img)  {

        mv_image_t* smaller = mv_create_image(mv_size_t(img->width / 2, img->height / 2),
            img->depth, img->nChannels);
        mv_resize_nn(img, smaller);

        return smaller;
    }


    // Detects features at extrema in DoG scale space.  Bad features are discarded
    // based on contrast and ratio of principal curvatures.
    int scale_space_extrema() {

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
                            WRITE_INFO_LOG("is_too_edge_like o=%d, i=%d, r=%d, c=%d", o, i, r, c);
                            rollback_new_feature();
                        }
                    }
                }
            }
        }
        return 0;
    }


    // Determines whether a pixel is a scale-space extremum by comparing it to it's
    // 3x3x3 pixel neighborhood.
    bool is_extremum(pyramid_layer* dogs, int r, int c)
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

    
    // Interpolates a scale-space extremum's location and scale to subpixel
    // accuracy to form an image feature.  Rejects features with low contrast.
    // Based on Section 4 of Lowe's paper.
    struct feature* interp_extremum(pyramid_layer* dogs, int octv, int intvl, int r, int c)
    {        
        double xi, xr, xc;        
        Vector3d X;
        Vector3d dD;
        Matrix3d H;

        int i = 0;
        for (; i < SIFT_MAX_INTERP_STEPS; i++) {
            
            // Performs one step of extremum interpolation.  Based on Eqn. (3) in Lowe's paper.
            //interp_step(dogs, r, c, X);
            deriv_3D(dogs, r, c, dD);
            hessian_3D(dogs, r, c, H);
            X = H.inverse() * (-1) * dD;
            
            xc = X(0); xr = X(1); xi = X(2);            
            if (ABS(xi) < 0.5  &&  ABS(xr) < 0.5  &&  ABS(xc) < 0.5)
                break;

            c += mv_round(xc);
            r += mv_round(xr);
            intvl += mv_round(xi);

            if (intvl < 1 || intvl > SIFT_INTVLS)
                return NULL;
            
            if (c < SIFT_IMG_BORDER || c >= dogs->curr->width - SIFT_IMG_BORDER)
                return NULL;

            if (r < SIFT_IMG_BORDER || r >= dogs->curr->height - SIFT_IMG_BORDER)            
                return NULL;                        
        }

        /* ensure convergence of interpolation */
        if (i >= SIFT_MAX_INTERP_STEPS)
            return NULL;

        //  Calculates interpolated pixel contrast.  Based on Eqn. (3) in Lowe's paper.
        //double contr = interp_contr(dogs, r, c, X);        
        deriv_3D(dogs, r, c, dD);
        MatrixXd T = dD.transpose() * X;
        double contr = pixval32f(dogs->curr, r, c) + T(0, 0) * 0.5;
        
        if (ABS(contr) < SIFT_CONTR_THR / SIFT_INTVLS)
            return NULL;

        feature* feat = new_feature();
        detection_data* ddata = &feat->ddata;
        feat->img_pt.x = feat->x = (c + xc) * pow(2.0, octv);
        feat->img_pt.y = feat->y = (r + xr) * pow(2.0, octv);
        ddata->r = r;
        ddata->c = c;
        ddata->octv = octv;
        ddata->intvl = intvl;
        ddata->subintvl = xi;

        return feat;
    }

    
    // Computes the partial derivatives in x, y, and scale of a pixel in the DoG scale space pyramid.
    void deriv_3D(pyramid_layer* dogs, int r, int c, Vector3d& result) {
        result(0) = (pixval32f(dogs->curr, r, c + 1) - pixval32f(dogs->curr, r, c - 1)) / 2.0;
        result(1) = (pixval32f(dogs->curr, r + 1, c) - pixval32f(dogs->curr, r - 1, c)) / 2.0;
        result(2) = (pixval32f(dogs->next, r, c) - pixval32f(dogs->prev, r, c)) / 2.0;
    }


    // Computes the 3D Hessian matrix for a pixel in the DoG scale space pyramid.
    void hessian_3D(pyramid_layer* dogs, int r, int c, Matrix3d& H) {
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

        H << dxx, dxy, dxs,
            dxy, dyy, dys,
            dxs, dys, dss;
    }

       
    struct feature* new_feature()
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

    void rollback_new_feature() {
        if (m_pool_used > 0) {
            m_pool_used--;
        }
    }


    // Determines whether a feature is too edge like to be stable by computing the
    // ratio of principal curvatures at that feature.  Based on Section 4.1 of Lowe's paper.
    bool is_too_edge_like(mv_image_t* dog, int r, int c)
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


    // Calculates characteristic scale for each feature in an array.
    void calc_feature_scales() {
        struct feature* feat;
        struct detection_data* ddata;
        double intvl;

        int n = m_pool_used;
        for (int i = 0; i < n; i++) {
            feat = m_pool + i;
            ddata = &feat->ddata;
            intvl = ddata->intvl + ddata->subintvl;
            feat->scl = SIFT_SIGMA * pow(2.0, ddata->octv + intvl / SIFT_INTVLS);
            ddata->scl_octv = SIFT_SIGMA * pow(2.0, intvl / SIFT_INTVLS);
        }
    }

    
    // Halves feature coordinates and scale in case the input image was doubled prior to scale space construction.
    void adjust_for_img_dbl() {
        struct feature* feat;

        int n = m_pool_used;
        for (int i = 0; i < n; i++) {
            feat = m_pool + i;
            feat->x /= 2.0;
            feat->y /= 2.0;
            feat->scl /= 2.0;
            feat->img_pt.x /= 2.0;
            feat->img_pt.y /= 2.0;
        }
    }


    // Computes a canonical orientation for each image feature in an array.  Based on Section 5 of Lowe's paper.  
    // This function adds features to the array when there is more than one dominant orientation at a given feature location.
    void calc_feature_oris() {
        struct feature* feat;
        struct detection_data* ddata;
        double hist[SIFT_ORI_HIST_BINS];

        int n = m_pool_used;
        for (int i = 0; i < n; i++) {
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

    
    // Computes a gradient orientation histogram at a specified pixel.
    void ori_hist(double* hist, mv_image_t* img, int r, int c, int rad, double sigma) {
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

    
    // Calculates the gradient magnitude and orientation at a given pixel.
    bool calc_grad_mag_ori(mv_image_t* img, int r, int c, double* mag, double* ori) {

        if (r > 0 && r < img->height - 1 && c > 0 && c < img->width - 1) {
            double dx = pixval32f(img, r, c + 1) - pixval32f(img, r, c - 1);
            double dy = pixval32f(img, r - 1, c) - pixval32f(img, r + 1, c);
            *mag = sqrt(dx*dx + dy*dy);
            *ori = atan2(dy, dx);
            return true;
        }
        
        return false;
    }

    
    // Gaussian smooths an orientation histogram.
    void smooth_ori_hist(double* hist) {
        double tmp, h0 = hist[0];
        double prev = hist[SIFT_ORI_HIST_BINS - 1];

        for (int i = 0; i < SIFT_ORI_HIST_BINS; i++) {
            tmp = hist[i];
            hist[i] = 0.25 * prev + 0.5 * hist[i] + 0.25 * ((i + 1 == SIFT_ORI_HIST_BINS) ? h0 : hist[i + 1]);
            prev = tmp;
        }
    }
    
    // Finds the magnitude of the dominant orientation in a histogram
    double dominant_ori(double* hist) {        
        int maxbin = 0;
        double omax = hist[0];        
        for (int i = 1; i < SIFT_ORI_HIST_BINS; i++) {
            if (hist[i] > omax) {
                omax = hist[i];
                maxbin = i;
            }
        }
        return omax;
    }

    
    // Adds features to an array for every orientation in a histogram greater than
    //  a specified threshold.
    void adjust_good_ori_features(double* hist, double mag_thr, feature* feat) {
        
        for (int i = 0; i < SIFT_ORI_HIST_BINS; i++) {
            int l = (i == 0) ? SIFT_ORI_HIST_BINS - 1 : i - 1;
            int r = (i + 1) % SIFT_ORI_HIST_BINS;

            if (hist[i] > hist[l] && hist[i] > hist[r] && hist[i] >= mag_thr)
            {                
                //Interpolates a histogram peak from left, center, and right values
                double bin = i + (0.5 * (hist[l] - hist[r]) / (hist[l] - 2.0 * hist[i] + hist[r]));
                bin = (bin < 0) ? SIFT_ORI_HIST_BINS + bin : (bin >= SIFT_ORI_HIST_BINS) ? bin - SIFT_ORI_HIST_BINS : bin;                
                feat->ori = ((MV_PI2 * bin) / SIFT_ORI_HIST_BINS) - MV_PI;                
            }
        }
    }


    // Computes feature descriptors for features in an array.  Based on Section 6 of Lowe's paper.    
    void compute_descriptors() {
        struct feature* feat;
        struct detection_data* ddata;
        double hist[SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS];

        int k = m_pool_used;
        for (int i = 0; i < k; i++) {
            feat = m_pool + i;
            ddata = &feat->ddata;

            mv_image_t* gauss = gauss_pyramid(ddata->octv, ddata->intvl);
            descr_hist(hist, gauss, ddata->r, ddata->c, feat->ori, ddata->scl_octv);
            hist_to_descr(hist, feat);
        }
    }

    
    // Computes the 2D array of orientation histograms that form the feature descriptor.  
    // Based on Section 6.1 of Lowe's paper.
    void descr_hist(double* hist, mv_image_t* img, int r, int c, double ori, double scl) {
        double grad_mag;
        double grad_ori;

        double cos_t = cos(ori);
        double sin_t = sin(ori);
        double bins_per_rad = SIFT_DESCR_HIST_BINS / MV_PI2;
        double exp_denom = SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * 0.5;
        double hist_width = SIFT_DESCR_SCL_FCTR * scl;
        int radius = mv_floor(hist_width * sqrt(2.0) * (SIFT_DESCR_WIDTH + 1.0) * 0.5 + 0.5);

        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
            
                //  Calculate sample's histogram array coords rotated relative to ori.
                //  Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e. r_rot = 1.5) 
                // have full weight placed in row 1 after interpolation.              
                double c_rot = (j * cos_t - i * sin_t) / hist_width;
                double r_rot = (j * sin_t + i * cos_t) / hist_width;
                double rbin = r_rot + SIFT_DESCR_WIDTH / 2 - 0.5;
                double cbin = c_rot + SIFT_DESCR_WIDTH / 2 - 0.5;

                if (rbin > -1.0  &&  rbin < SIFT_DESCR_WIDTH  &&  cbin > -1.0  &&  cbin < SIFT_DESCR_WIDTH) {
                    if (calc_grad_mag_ori(img, r + i, c + j, &grad_mag, &grad_ori)) {
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
        }
    }

    
    //  Interpolates an entry into the array of orientation histograms that form the feature descriptor.
    void interp_hist_entry(double* hist, double rbin, double cbin, double obin, double mag)
    {
        double* pos;

        int r0 = mv_floor(rbin);
        int c0 = mv_floor(cbin);
        int o0 = mv_floor(obin);
        double d_r = rbin - r0;
        double d_c = cbin - c0;
        double d_o = obin - o0;

        
        // The entry is distributed into up to 8 bins.  Each entry into a bin is multiplied 
        //  by a weight of 1 - d for each dimension, where d is the distance from the center 
        //  value of the bin measured in bin units.          
        for (int r = 0; r <= 1; r++) {
            int rb = r0 + r;
            if (rb >= 0 && rb < SIFT_DESCR_WIDTH) {
                double v_r = mag * ((r == 0) ? 1.0 - d_r : d_r);
                //row = hist[rb];
                for (int c = 0; c <= 1; c++) {
                    int cb = c0 + c;
                    if (cb >= 0 && cb < SIFT_DESCR_WIDTH) {
                        double v_c = v_r * ((c == 0) ? 1.0 - d_c : d_c);
                        //h = row[cb];
                        for (int o = 0; o <= 1; o++) {
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

    
    // Converts the 2D array of orientation histograms into a feature's descriptor vector.
    void hist_to_descr(double* hist, struct feature* feat) {
        
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
            int_val = (int)(SIFT_INT_DESCR_FCTR * feat->descr[i]);
            feat->descr[i] = MIN(255, int_val);
        }
    }


    // Normalizes a feature's descriptor vector to unitl length
    void normalize_descr(struct feature* feat) {
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

    
    // Compares features for a decreasing-scale ordering.  
    static int feature_cmp(const void* feat1, const void* feat2) {
        struct feature* f1 = (struct feature*) feat1;
        struct feature* f2 = (struct feature*) feat2;

        if (f1->scl < f2->scl)
            return 1;
        if (f1->scl > f2->scl)
            return -1;

        return 0;
    }


    // De-allocates memory held by a scale space pyramid
    void release_pyramid() {        
        for (int o = 0; o < MAX_OCTVS; o++)  {
            for (int i = 0; i < SIFT_INTVLS + 3; i++)  {

                if (m_gauss_pyr[o][i] != NULL) {
                    mv_release_image(&m_gauss_pyr[o][i]);
                }

                if (m_dog_pyr[o][i] != NULL) {
                    mv_release_image(&m_dog_pyr[o][i]);
                }
            }
        }  
    }



    void export_features(mv_features* features) {
        feature* items[MAX_FEATURE_SIZE];
        for (int i = 0; i < m_pool_used; i++) {
            items[i] = m_pool + i;
        }

        qsort(items, m_pool_used, sizeof(feature*), feature_cmp);

        for (int i = 0; i < m_pool_used; i++) {
            features->push_back(items[i]);
        }
    }

};



sift_module* sift_module::create_instance()
{
    return new sift_runtime();
}

