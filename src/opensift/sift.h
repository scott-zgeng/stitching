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

static const int MAX_FEATURE_SIZE = 1024 * 2;
typedef mv_vector<feature*, MAX_FEATURE_SIZE> mv_features;


class sift_module
{ 
public:
    virtual ~sift_module() = 0;
    virtual int process(mv_image_t* img, mv_features* features) = 0;

public:
    static sift_module* create_instance();
};



#endif
