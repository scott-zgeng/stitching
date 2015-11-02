/**@file
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

#ifndef SIFT_H
#define SIFT_H

#include "../img_proc.h"
#include "../mv_vector.h"

#include "imgfeatures.h"

class sift_runtime
{
public:
    struct pyramid_layer
    {
        mv_image_t* curr;
        mv_image_t* prev;
        mv_image_t* next;
    };

    static const int MAX_FEATURE_SIZE = 1024 * 2;  // 最多保存的特征数

    /** default number of sampled intervals per octave */
    static const int SIFT_INTVLS = 3;

    /** default sigma for initial gaussian smoothing */
    const double SIFT_SIGMA = 1.6;

    /** default threshold on keypoint contrast |D(x)| */
    const double SIFT_CONTR_THR = 0.04;

    /** default threshold on keypoint ratio of principle curvatures */
    static const int SIFT_CURV_THR = 10;

    /** double image size before pyramid construction? */
    static const int SIFT_IMG_DBL = 1;

    /** default width of descriptor histogram array */
    static const int SIFT_DESCR_WIDTH = 4;

    /** default number of bins per histogram in descriptor array */
    static const int SIFT_DESCR_HIST_BINS = 8;

    /* assumed gaussian blur for input image */
    const double SIFT_INIT_SIGMA = 0.5;

    /* width of border in which to ignore keypoints */
    static const int SIFT_IMG_BORDER = 5;

    /* maximum steps of keypoint interpolation before failure */
    static const int SIFT_MAX_INTERP_STEPS = 5;

    /* default number of bins in histogram for orientation assignment */
    static const int SIFT_ORI_HIST_BINS = 36;

    /* determines gaussian sigma for orientation assignment */
    const double SIFT_ORI_SIG_FCTR = 1.5;

    /* determines the radius of the region used in orientation assignment */
    const double SIFT_ORI_RADIUS = 3.0 * SIFT_ORI_SIG_FCTR;

    /* number of passes of orientation histogram smoothing */
    static const int SIFT_ORI_SMOOTH_PASSES = 2;

    /* orientation magnitude relative to max that results in new feature */
    const double SIFT_ORI_PEAK_RATIO = 0.8;

    /* determines the size of a single descriptor orientation histogram */
    const double SIFT_DESCR_SCL_FCTR = 3.0;

    /* threshold on magnitude of elements of descriptor vector */
    const double SIFT_DESCR_MAG_THR = 0.2;

    /* factor used to convert floating-point descriptor to unsigned char */
    const double SIFT_INT_DESCR_FCTR = 512.0;

    static const int MAX_OCTVS = 10; 

    const double PRELIM_CONTR_THR = 0.5 * SIFT_CONTR_THR / SIFT_INTVLS;


public:
    sift_runtime();
    virtual ~sift_runtime();

public:   
    typedef public mv_vector<feature*, MAX_FEATURE_SIZE> mv_features;

    /**
    Finda SIFT features in an image using user-specified parameter values.  All
    detected features are stored in the array pointed to by \a feat.

    @param img the image in which to detect features
    @param feat a pointer to an array in which to store detected features;
    memory for this array is allocated by this function and must be freed by
    the caller using free(*feat)
    @param intvls the number of intervals sampled per octave of scale space
    @param sigma the amount of Gaussian smoothing applied to each image level
    before building the scale space representation for an octave
    @param contr_thr a threshold on the value of the scale space function
    \f$\left|D(\hat{x})\right|\f$, where \f$\hat{x}\f$ is a vector specifying
    feature location and scale, used to reject unstable features;  assumes
    pixel values in the range [0, 1]
    @param curv_thr threshold on a feature's ratio of principle curvatures
    used to reject features that are too edge-like
    @param img_dbl should be 1 if image doubling prior to scale space
    construction is desired or 0 if not
    @param descr_width the width, \f$n\f$, of the \f$n \times n\f$ array of
    orientation histograms used to compute a feature's descriptor
    @param descr_hist_bins the number of orientations in each of the
    histograms in the array used to compute a feature's descriptor

    @return Returns the number of keypoints stored in \a feat or -1 on failure    
    */

    int process(mv_image_t* img, mv_features* features);

private:

    inline mv_image_t* gauss_pyramid(int octvs, int intval) {
        return m_gauss_pyr[octvs * m_octvs + intval];
    }

    inline void set_gauss_pyramid(mv_image_t* img, int octvs, int intval) {
        m_gauss_pyr[octvs * m_octvs + intval] = img;
    }

    inline mv_image_t* dog_pyramid(int octvs, int intval) {
        return m_dog_pyr[octvs * m_octvs + intval];
    }

    inline void set_dog_pyramid(mv_image_t* img, int octvs, int intval) {
        m_dog_pyr[octvs * m_octvs + intval] = img;
    }

private:
    mv_image_t* create_init_img(mv_image_t* img, bool is_double);
    
    int build_gauss_pyramid(mv_image_t* base);
    int build_dog_pyramid();

    mv_image_t* sift_runtime::downsample(mv_image_t* img);
    int scale_space_extrema();
    
    bool is_extremum(pyramid_layer* dogs, int r, int c);
    struct feature* interp_extremum(pyramid_layer* dogs, int octv, int intvl, int r, int c);
    void interp_step(pyramid_layer* dogs, int r, int c, double* xi, double* xr, double* xc);       
    
    mv_mat_handle hessian_3D(pyramid_layer* dogs, int r, int c);
    void deriv_3D(pyramid_layer* dogs, int r, int c, double* result);
    double interp_contr(pyramid_layer* dogs, int r, int c, double xi, double xr, double xc);           
    
    struct feature* new_feature();
    void release_last_feature();
    bool is_too_edge_like(mv_image_t* dog, int r, int c);

    void calc_feature_scales();
    void adjust_for_img_dbl();

    void ori_hist(double* hist, mv_image_t* img, int r, int c, int rad, double sigma);
    
    void calc_feature_oris();

    int calc_grad_mag_ori(mv_image_t* img, int r, int c, double* mag, double* ori);
    void smooth_ori_hist(double* hist);
    double dominant_ori(double* hist);

    void adjust_good_ori_features(double* hist, double mag_thr, feature* feat);
    
    void compute_descriptors();
    void descr_hist(double* hist, mv_image_t* img, int r, int c, double ori, double scl);
    void interp_hist_entry(double* hist, double rbin, double cbin, double obin, double mag);    
    void hist_to_descr(double* hist, struct feature* feat);

    void normalize_descr(struct feature*);
    static int feature_cmp(const void* feat1, const void* feat2);
    void release_pyramid();

    void export(mv_features* features);

private:
    feature m_pool[MAX_FEATURE_SIZE];    
    int m_pool_used;

    int m_octvs;
    mv_image_t* m_gauss_pyr[MAX_OCTVS * SIFT_INTVLS + MAX_OCTVS * 3];
    mv_image_t* m_dog_pyr[MAX_OCTVS * SIFT_INTVLS + MAX_OCTVS * 3];
};



#endif
