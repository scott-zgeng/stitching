/**@file
   Functions for computing transforms from image feature correspondences.

   Copyright (C) 2006-2012  Rob Hess <rob@iqengines.com>

   @version 1.1.2-20100521
   */

#ifndef XFORM_H
#define XFORM_H


#include "../mv_base.h"
#include "sift.h"

#include <Eigen/Dense>
using namespace Eigen;

struct feature;

class ransac_module
{

public:
    virtual ~ransac_module() = 0;
    virtual int process(mv_features* features, Matrix3d& H) = 0;
    virtual void export_inlier(feature*** inliers, int* n) = 0;

    static ransac_module* instance();
};


#endif
