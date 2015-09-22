/**@file
   Functions and structures for dealing with image features.

   Copyright (C) 2006-2012  Rob Hess <rob@iqengines.com>

   @version 1.1.2-20100521
   */

#ifndef IMGFEATURES_H
#define IMGFEATURES_H

#include "../mv_base.h"




/** FEATURE_OXFD <BR> FEATURE_LOWE */
enum feature_type
{
    FEATURE_OXFD,
    FEATURE_LOWE,
};

/** FEATURE_FWD_MATCH <BR> FEATURE_BCK_MATCH <BR> FEATURE_MDL_MATCH */
enum feature_match_type
{
    FEATURE_FWD_MATCH,
    FEATURE_BCK_MATCH,
    FEATURE_MDL_MATCH,
};


/* colors in which to display different feature types */
#define FEATURE_OXFD_COLOR MV_RGB(255,255,0)
#define FEATURE_LOWE_COLOR MV_RGB(255,0,255)

/** max feature descriptor length */
#define FEATURE_MAX_D 128


/**
   Structure to represent an affine invariant image feature.  The fields
   x, y, a, b, c represent the affine region around the feature:

   a(x-u)(x-u) + 2b(x-u)(y-v) + c(y-v)(y-v) = 1
   */
struct feature
{
    double x;                      /**< x coord */
    double y;                      /**< y coord */
    double a;                      /**< Oxford-type affine region parameter */
    double b;                      /**< Oxford-type affine region parameter */
    double c;                      /**< Oxford-type affine region parameter */
    double scl;                    /**< scale of a Lowe-style feature */
    double ori;                    /**< orientation of a Lowe-style feature */
    int d;                         /**< descriptor length */
    double descr[FEATURE_MAX_D];   /**< descriptor */
    int type;                      /**< feature type, OXFD or LOWE */
    int category;                  /**< all-purpose feature category */
    struct feature* fwd_match;     /**< matching feature from forward image */
    struct feature* bck_match;     /**< matching feature from backmward image */
    struct feature* mdl_match;     /**< matching feature from model */
    mv_point_d_t img_pt;           /**< location in image */
    mv_point_d_t mdl_pt;           /**< location in model */
    void* feature_data;            /**< user-definable data */
};



/**
   Displays a set of features on an image

   @param img image on which to display features
   @param feat array of Oxford-type features
   @param n number of features
   */
extern void draw_features(mv_image_t* img, struct feature* feat, int n);


/**
   Calculates the squared Euclidian distance between two feature descriptors.

   @param f1 first feature
   @param f2 second feature

   @return Returns the squared Euclidian distance between the descriptors of
   \a f1 and \a f2.
   */
extern double descr_dist_sq(struct feature* f1, struct feature* f2);


#endif
