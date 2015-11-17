/*
  This file contains definitions for functions to compute transforms from
  image feature correspondences

  Copyright (C) 2006-2012  Rob Hess <rob@iqengines.com>

  @version 1.1.2-20100521
  */


#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>


#include "xform.h"
#include "imgfeatures.h"
#include "utils.h"

//#include <Eigen/Dense>
//using namespace Eigen;



///* extracts a feature's RANSAC data */
//#define feat_ransac_data( feat ) ( (struct ransac_data*) (feat)->feature_data )




class ransac_impl : public ransac_module
{
public:
    // RANSAC error tolerance in pixels 
    static const int RANSAC_ERR_TOL = 3;

    static const int RANSAC_XFORM_MIN_SIZE = 4;

    // pessimistic estimate of fraction of inlers for RANSAC
    const double RANSAC_INLIER_FRAC_EST = 0.25;

    // estimate of the probability that a correspondence supports a bad model 
    const double RANSAC_PROB_BAD_SUPP = 0.10;

    const double RANSAC_BAD_XFORM = 0.01;

    const double RANSAC_TOTAL_ERROR = 3.0;

    // holds feature data relevant to ransac 
    struct ransac_data {
        feature* feat;
        bool sampled;
    };


private:    
    ransac_data* m_match_list[MAX_FEATURE_SIZE];
    int m_match_num;
    feature* m_consensus_buf[2][MAX_FEATURE_SIZE];    
    feature** m_consensus;
    int m_consensus_num;
    int m_in_min;

public:
    ransac_impl() {
        srand((int)time(NULL));
        m_match_num = 0;    
        m_consensus = NULL;
        m_consensus_num = 0;

        m_in_min = calc_min_inliers();
    }

    virtual ~ransac_impl() {

    }

    // Calculates a best-fit image transform from image feature correspondences using RANSAC.    
    virtual int process(mv_features* features, Matrix3d& H) {
        assert(features->size() <= MAX_FEATURE_SIZE);
                
        int ret = get_matched_features(features);
        if (ret != 0) return -1;     

        int in_min = m_in_min;
        int in_max = 0;
        int in, k = 0;
        double in_frac = RANSAC_INLIER_FRAC_EST;
        double p = pow(1.0 - pow(in_frac, RANSAC_XFORM_MIN_SIZE), k);

        int flip_flop = 0;
        feature* samples[RANSAC_XFORM_MIN_SIZE];        
        feature** consensus;
        feature** consensus_max = NULL;

        while (p > RANSAC_BAD_XFORM) {
            draw_ransac_sample(samples, RANSAC_XFORM_MIN_SIZE);            
            lsq_homog(samples, RANSAC_XFORM_MIN_SIZE, H);
            
            consensus = m_consensus_buf[flip_flop];
            in = find_consensus(H, consensus);
            if (in > in_max) {                
                consensus_max = consensus;
                in_max = in;
                in_frac = (double)in_max / m_match_num;
                flip_flop = !flip_flop;
            }            
            
            p = pow(1.0 - pow(in_frac, RANSAC_XFORM_MIN_SIZE), ++k);
        }

        // calculate final transform based on best consensus set 
        if (in_max >= in_min) {            
            lsq_homog(consensus_max, in_max, H);

            consensus = m_consensus_buf[flip_flop];
            in = find_consensus(H, consensus);
            lsq_homog(consensus, in, H);

            m_consensus = consensus;
            m_consensus_num = in;
        } 
       
        return 0;
    }

    virtual void export_inlier(feature*** inliers, int* n) {
        *inliers = m_consensus;
        *n = m_consensus_num;
    }

private:


    // Calculates a least-squares planar homography from point correspondeces.
    void lsq_homog(feature* features[], int n, Matrix3d& H)
    {
        // set up matrices so we can unstack homography into X; AX = B 
        MatrixXd A(2 * n, 8);
        MatrixXd B(2 * n, 1);
        Matrix<double, 8, 1> X;
        
        A.setZero();
        mv_point_d_t* pt;
        mv_point_d_t* mpt;
        
        for (int i = 0; i < n; i++) {
            pt = &features[i]->img_pt;            
            mpt = &(get_match(features[i])->img_pt);

            A(i, 0) = pt->x;
            A(i + n, 3) = pt->x;
            A(i, 1) = pt->y;
            A(i + n, 4) = pt->y;
            A(i, 2) = 1.0;
            A(i + n, 5) = 1.0;
            A(i, 6) = -pt->x * mpt->x;
            A(i, 7) = -pt->y * mpt->x;
            A(i + n, 6) = -pt->x * mpt->y;
            A(i + n, 7) = -pt->y * mpt->y;

            B(i, 0) = mpt->x;
            B(i + n, 0) = mpt->y;
        }

        JacobiSVD<MatrixXd> svd(A);
        X = svd.solve(B);        

        H <<X(0), X(1), X(2),
            X(3), X(4), X(5),
            X(6), X(7), 1.0;        
    }


    
    // Calculates the transfer error between a point and its correspondence for
    // a given homography, i.e. for a point x, it's correspondence x', and
    // homography H, computes d(x', Hx)^2.
    double homog_xfer_err(mv_point_d_t pt, mv_point_d_t mpt, const Matrix3d& H) {
        
        // Performs a perspective transformation on a single point.  That is, for a
        // point (x, y) and a 3 x 3 matrix T this function returns the point (u, v), where
        //  [x' y' w']^T = T * [x y 1]^T,       
        //      and
        //  (u, v) = (x'/w', y'/w').
        //  Note that affine transforms are a subset of perspective transforms.
        
        Vector3d XY;
        XY << pt.x, pt.y, 1.0;

        Vector3d UY;
        UY = H * XY;

        mv_point_d_t xpt(UY(0) / UY(2), UY(1) / UY(2));

        return sqrt(dist_sq_2D(xpt, mpt));
    }


    // Returns a feature's match according to a specified match type
    inline struct feature* get_match(struct feature* feat) {
        return feat->fwd_match;        
    }


    // Finds all features with a match of a specified type and stores pointers
    // to them in an array.  Additionally initializes each matched feature's
    // feature_data field with a ransac_data structure.
    int get_matched_features(mv_features* features) {
        int m = 0;
        int n = features->size();
        for (int i = 0; i < n; i++) {
            feature* matched = get_match(features->at(i));
            if (matched == NULL)
                continue;

            m_match_list[m]->feat = features->at(i);
            m_match_list[m]->sampled = false;
            m++;
        }

        m_match_num = m;       

        if (m < RANSAC_XFORM_MIN_SIZE) {
            WRITE_ERROR_LOG("not enough matches to compute xform.");
            return -1;
        }

        return 0;
    }



    
    // Calculates the minimum number of inliers as a function of the number of putative correspondences.      
    int calc_min_inliers() {
        int num = m_match_num;
        double p_badsupp = RANSAC_PROB_BAD_SUPP;
        double p_badxform = RANSAC_BAD_XFORM;
        int m = RANSAC_XFORM_MIN_SIZE;

        double sum;        
        int n = 0;        

        for (int n = m + 1; n <= num; n++) {
            sum = 0;
            for (int i = n; i <= num; i++) {
                double pi = (i - m) * log(p_badsupp) + (num - i + m) * log(1.0 - p_badsupp) +
                    log_factorial(num - m) - log_factorial(i - m) - log_factorial(num - i);
                
                // Last three terms above are equivalent to log( n-m choose i-m )                
                sum += exp(pi);
            }

            if (sum < p_badxform)
                break;
        }
        return n;
    }


    // Calculates the natural log of the factorial of a number
    inline double log_factorial(int n) {
        double f = 0;
        int i;

        for (i = 1; i <= n; i++)
            f += log((double)i);

        return f;
    }

    
    // Draws a RANSAC sample from a set of features.
    void draw_ransac_sample(feature* samples[], int n) {        
        ransac_data* rdata;
                
        for (int i = 0; i < m_match_num; i++) {
            do {     
                rdata = m_match_list[rand() % n];
                if (!rdata->sampled) break;                                
            } while (true);

            samples[i] = rdata->feat;
            rdata->sampled = true;
        }        
    }

    
    // For a given model and error function, finds a consensus from a set of feature correspondences.
    int find_consensus(const Matrix3d& M, feature* consensus[]) {
        feature* feat;
        feature* match;
        mv_point_d_t pt, mpt;
        
        int n = 0; 
        for (int i = 0; i < m_match_num; i++) {
            feat = m_match_list[i]->feat;
            match = get_match(feat);

            pt = feat->img_pt;
            mpt = match->img_pt;            
            if (homog_xfer_err(pt, mpt, M) <= RANSAC_TOTAL_ERROR)
                consensus[n++] = feat;
        }
        return n;
    }

};


ransac_module* ransac_module::instance()
{
    ransac_impl inst;
    return &inst;
}

