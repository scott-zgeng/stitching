/*
  Functions and structures for dealing with image features

  Copyright (C) 2006-2012  Rob Hess <rob@iqengines.com>

  @version 1.1.2-20100521
  */


#include <math.h>
#include <float.h>

#include "utils.h"
#include "imgfeatures.h"



//static void draw_oxfd_features(mv_image_t*, struct feature*, int);
//static void draw_oxfd_feature(mv_image_t*, struct feature*, mv_scalar_t);


//static void draw_lowe_features(mv_image_t*, struct feature*, int);
//static void draw_lowe_feature(mv_image_t*, struct feature*, mv_scalar_t);





/*
  Draws a set of features on an image

  @param img image on which to draw features
  @param feat array of Oxford-type features
  @param n number of features
  */
//void draw_features(mv_image_t* img, struct feature* feat, int n)
//{
//    int type;
//
//    if (n <= 0 || !feat) {
//        WRITE_WARN_LOG("Warning: no features to draw");            
//        return;
//    }
//
//    type = feat[0].type;
//    switch (type)
//    {
//    case FEATURE_OXFD:
//        draw_oxfd_features(img, feat, n);
//        break;
//    case FEATURE_LOWE:
//        draw_lowe_features(img, feat, n);
//        break;
//    default:
//        WRITE_WARN_LOG("Warning: draw_features(): unrecognized feature");            
//        break;
//    }
//}



/*
  Calculates the squared Euclidian distance between two feature descriptors.

  @param f1 first feature
  @param f2 second feature

  @return Returns the squared Euclidian distance between the descriptors of
  f1 and f2.
  */
double descr_dist_sq(struct feature* f1, struct feature* f2)
{
    double diff, dsq = 0;
    double* descr1, *descr2;
    int i, d;

    d = f1->d;
    if (f2->d != d)
        return DBL_MAX;
    descr1 = f1->descr;
    descr2 = f2->descr;

    for (i = 0; i < d; i++)
    {
        diff = descr1[i] - descr2[i];
        dsq += diff*diff;
    }
    return dsq;
}




/*
  Draws Oxford-type affine features

  @param img image on which to draw features
  @param feat array of Oxford-type features
  @param n number of features
  */
//static void draw_oxfd_features(mv_image_t* img, struct feature* feat, int n)
//{
//    mv_scalar_t color = MV_RGB(255, 255, 255);
//    int i;
//
//    if (img->nChannels > 1)
//        color = FEATURE_OXFD_COLOR;
//    for (i = 0; i < n; i++)
//        draw_oxfd_feature(img, feat + i, color);
//}
//


/*
  Draws a single Oxford-type feature

  @param img image on which to draw
  @param feat feature to be drawn
  @param color color in which to draw
  */
//static void draw_oxfd_feature(mv_image_t* img, struct feature* feat,
//    mv_scalar_t color)
//{
//    double m[4] = { feat->a, feat->b, feat->b, feat->c };
//    double v[4] = { 0 };
//    double e[2] = { 0 };
//    mv_matrix_t M, V, E;
//    double alpha, l1, l2;
//
//    /* compute axes and orientation of ellipse surrounding affine region */
//    mv_init_matrix_header(&M, 2, 2, MV_64FC1, m, MV_AUTOSTEP);
//    mv_init_matrix_header(&V, 2, 2, MV_64FC1, v, MV_AUTOSTEP);
//    mv_init_matrix_header(&E, 2, 1, MV_64FC1, e, MV_AUTOSTEP);
//    mv_eigen_val_vector(&M, &V, &E, DBL_EPSILON, 0, 0);
//
//    l1 = 1 / sqrt(e[1]);
//    l2 = 1 / sqrt(e[0]);
//    alpha = -atan2(v[1], v[0]);
//    alpha *= 180 / MV_M_PI;
//
//    mv_ellipse(img, mv_point_t(feat->x, feat->y), mv_size_t(l2, l1), alpha,
//        0, 360, MV_RGB(0, 0, 0), 3, 8, 0);
//    mv_ellipse(img, mv_point_t(feat->x, feat->y), mv_size_t(l2, l1), alpha,
//        0, 360, color, 1, 8, 0);
//    mv_line(img, mv_point_t(feat->x + 2, feat->y), mv_point_t(feat->x - 2, feat->y),
//        color, 1, 8, 0);
//    mv_line(img, mv_point_t(feat->x, feat->y + 2), mv_point_t(feat->x, feat->y - 2),
//        color, 1, 8, 0);
//}


/*
  Draws Lowe-type features

  @param img image on which to draw features
  @param feat array of Oxford-type features
  @param n number of features
  */
//static void draw_lowe_features(mv_image_t* img, struct feature* feat, int n)
//{
//    mv_scalar_t color = MV_RGB(255, 255, 255);
//    int i;
//
//    if (img->nChannels > 1)
//        color = FEATURE_LOWE_COLOR;
//    for (i = 0; i < n; i++)
//        draw_lowe_feature(img, feat + i, color);
//}
//

/*
Draws a single Lowe-type feature

@param img image on which to draw
@param feat feature to be drawn
@param color color in which to draw
*/
//static void draw_lowe_feature(mv_image_t* img, struct feature* feat,
//    mv_scalar_t color)
//{
//    int len, hlen, blen, start_x, start_y, end_x, end_y, h1_x, h1_y, h2_x, h2_y;
//    double scl, ori;
//    double scale = 5.0;
//    double hscale = 0.75;
//    mv_point_t start, end, h1, h2;
//
//    /* compute points for an arrow scaled and rotated by feat's scl and ori */
//    start_x = mv_round(feat->x);
//    start_y = mv_round(feat->y);
//    scl = feat->scl;
//    ori = feat->ori;
//    len = mv_round(scl * scale);
//    hlen = mv_round(scl * hscale);
//    blen = len - hlen;
//    end_x = mv_round(len *  cos(ori)) + start_x;
//    end_y = mv_round(len * -sin(ori)) + start_y;
//    h1_x = mv_round(blen *  cos(ori + MV_PI / 18.0)) + start_x;
//    h1_y = mv_round(blen * -sin(ori + MV_PI / 18.0)) + start_y;
//    h2_x = mv_round(blen *  cos(ori - MV_PI / 18.0)) + start_x;
//    h2_y = mv_round(blen * -sin(ori - MV_PI / 18.0)) + start_y;
//    start = mv_point_t(start_x, start_y);
//    end = mv_point_t(end_x, end_y);
//    h1 = mv_point_t(h1_x, h1_y);
//    h2 = mv_point_t(h2_x, h2_y);
//
//    mv_line(img, start, end, color, 1, 8, 0);
//    mv_line(img, end, h1, color, 1, 8, 0);
//    mv_line(img, end, h2, color, 1, 8, 0);
//}
