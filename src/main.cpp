/*
  Detects SIFT features in two images and finds matches between them.

  Copyright (C) 2006-2012  Rob Hess <rob@iqengines.com>

  @version 1.1.2-20100521
  */

#include "opensift/sift.h"
#include "opensift/imgfeatures.h"
#include "opensift/kdtree.h"
#include "opensift/utils.h"
#include "opensift/xform.h"



#include <stdio.h>

#include "highgui.h"


#include "mv_base.h"
#include "img_proc.h"


#pragma comment(lib, "opencv_core300d.lib")
#pragma comment(lib, "opencv_highgui300d.lib")
#pragma comment(lib, "opencv_imgcodecs300d.lib")
#pragma comment(lib, "opencv_imgproc300d.lib")


//�������ַ���
#define IMG1 "iamge1"
#define IMG2 "iamge2"
#define IMG1_FEAT "iamge1_feature"
#define IMG2_FEAT "iamge2_feature"
#define IMG_MATCH1 "match1"
#define IMG_MATCH2 "match2"
#define IMG_MOSAIC_TEMP "stitching_temp"
#define IMG_MOSAIC_SIMPLE "stitching_simple"
#define IMG_MOSAIC_BEFORE_FUSION "stitching_before_fusion"
#define IMG_MOSAIC_PROC "stitching"


/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

/* threshold on squared ratio of distances between NN and 2nd NN */
#define NN_SQ_DIST_RATIO_THR 0.49





void show_image(mv_image_t* img)
{
    const char* IMG_TEST_1 = "test_windows_1";
    IplImage* clone = mv_image_mv2cv(img);
    cvShowImage(IMG_TEST_1, clone);
    cvWaitKey(0);
    cvReleaseImage(&clone);
}


void show_cv_image(IplImage* img)
{
    const char* IMG_TEST_2 = "test_windows_2";
    cvShowImage(IMG_TEST_2, img);
    cvWaitKey(0);
}


void compare_image(mv_image_t* img1, IplImage* img2)
{
    int size = img1->width * img1->height;
    mv_byte* p1 = img1->imageData;
    mv_byte* p2 = (mv_byte*)img2->imageData;

    for (int i = 0; i < size; i++) {
        if (p1[i] != p2[i]) {
            printf("i = %d, p1 = %d, p2 = %d\n", i, p1[i], p2[i]);
            cvWaitKey(0);
        }
    }
}


class image_stitching
{
public:

    void calc_corner(const Matrix3d& H, mv_image_t* img2) {        
        Vector3d V1;
        Vector3d V2;        

        //��v2��������Ϊ���Ͻ�����
        V2 << 0, 0, 1;
        V1 = H * V2;
        m_left_top.x = mv_round(V1[0] / V1[2]);
        m_left_top.y = mv_round(V1[1] / V1[2]);

        //��v2��������Ϊ���½�����
        V2 << 0, img2->height, 1;
        V1 = H * V2;
        m_left_bottom.x = mv_round(V1[0] / V1[2]);
        m_left_bottom.y = mv_round(V1[1] / V1[2]);

        //��v2��������Ϊ���Ͻ�����
        V2 << img2->width, 0, 1;
        V1 = H * V2;
        m_right_top.x = mv_round(V1[0] / V1[2]);
        m_right_top.y = mv_round(V1[1] / V1[2]);
        
        //��v2��������Ϊ���½�����
        V2 << img2->width, img2->height, 1;
        V1 = H * V2;
        m_right_bottom.x = mv_round(V1[0] / V1[2]);
        m_right_bottom.y = mv_round(V1[1] / V1[2]);
    }

    mv_image_t* load_image(const char* filename) {
        IplImage* org = cvLoadImage(filename);
        if (org == NULL) {
            WRITE_ERROR_LOG("unable to load image from %s", filename);
            return NULL;
        }

        mv_image_t* img = mv_image_cv2mv(org);
        cvReleaseImage(&org);
        if (img == NULL) {            
            WRITE_ERROR_LOG("unable to load image from %s", filename);
            return NULL;
        }
        
        return img;
    }

    feature* features1[MAX_FEATURE_SIZE];
    feature* features2[MAX_FEATURE_SIZE];

    int image_stitching_entry()
    {
        WRITE_INFO_LOG("enter main function");

        const char* imgfile1 = "g1.jpg";
        const char* imgfile2 = "g2.jpg";

        //const char* imgfile1 = "c1.jpg";
        //const char* imgfile2 = "c2.jpg";

        mv_image_t* img1 = load_image(imgfile1);
        if (img1 == NULL) return -1;

        mv_image_t* img2 = load_image(imgfile2);
        if (img2 == NULL) return -1;


        //stacked = stack_imgs(img1, img2);

        WRITE_INFO_LOG("Finding features in %s...", imgfile1);
        sift_module* sift1 = sift_module::create_instance();        
        sift1->process(img1);

        //feature* features1[MAX_FEATURE_SIZE];
        int size1 = sift1->export_features(features1, MAX_FEATURE_SIZE);
        WRITE_INFO_LOG("Finding features %d", size1);

        sift_module* sift2 = sift_module::create_instance();        
        WRITE_INFO_LOG("Finding features in %s...", imgfile2);
        sift2->process(img2);

        //feature* features2[MAX_FEATURE_SIZE];
        int size2 = sift2->export_features(features2, MAX_FEATURE_SIZE);
        WRITE_INFO_LOG("Finding features %d", size2);
        
        struct feature** nbrs;
        
        mv_point_t pt1, pt2;
        double d0, d1;
        int n1, n2, k, m = 0;

        WRITE_INFO_LOG("Building kd tree...");
        kd_node* kd_root = kdtree_build(features1, size1);

        WRITE_INFO_LOG("kdtree_bbf_knn");
        for (int i = 0; i < size2; i++) {
            feature* feat = features2[i];
            int k = kdtree_bbf_knn(kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS);
            if (k == 2) {
                d0 = descr_dist_sq(feat, nbrs[0]);
                d1 = descr_dist_sq(feat, nbrs[1]);
                if (d0 < d1 * NN_SQ_DIST_RATIO_THR) {
                    pt1 = mv_point_t(mv_round(feat->x), mv_round(feat->y));
                    pt2 = mv_point_t(mv_round(nbrs[0]->x), mv_round(nbrs[0]->y));
                    pt2.y += img1->height;
                    //mv_line(stacked, pt1, pt2, MV_RGB(255, 0, 255), 1, 8, 0);
                    m++;
                    features2[i]->fwd_match = nbrs[0];
                }
            }
            free(nbrs);
        }

        WRITE_INFO_LOG("Found %d total matches", m);


        mv_image_t* xformed;
        struct feature **inliers;
        int n_inliers;

        Matrix3d H;
        int ret = ransac_module::instance()->process(features2, size2, H);
        if (ret != 0) {
            WRITE_ERROR_LOG("ransac_module failed");
            return -1;
        }

        ransac_module::instance()->export_inlier(&inliers, &n_inliers);
        WRITE_INFO_LOG("ransac_module inliers = %d", n_inliers);
        if (n_inliers == 0) {
            WRITE_ERROR_LOG("n_inliers == 0");
            return -1;
        }

        //���ܳɹ�������任���󣬼�����ͼ���й�ͬ����

        //stacked_ransac = stack_imgs_horizontal(img1, img2);//�ϳ�ͼ����ʾ��RANSAC�㷨ɸѡ���ƥ����

        //img1LeftBound = inliers[0]->fwd_match->x;//ͼ1��ƥ�����Ӿ��ε���߽�
        //img1RightBound = img1LeftBound;//ͼ1��ƥ�����Ӿ��ε��ұ߽�
        //img2LeftBound = inliers[0]->x;//ͼ2��ƥ�����Ӿ��ε���߽�
        //img2RightBound = img2LeftBound;//ͼ2��ƥ�����Ӿ��ε��ұ߽�

        int invertNum = 0;//ͳ��pt2.x > pt1.x��ƥ���Եĸ��������ж�img1���Ƿ���ͼ

        //������RANSAC�㷨ɸѡ��������㼯��inliers���ҵ�ÿ���������ƥ��㣬��������
        for (int i = 0; i < n_inliers; i++)
        {
            feature* feat = inliers[i];//��i��������
            mv_point_t pt2 = mv_point_t(mv_round(feat->x), mv_round(feat->y));//ͼ2�е������
            mv_point_t pt1 = mv_point_t(mv_round(feat->fwd_match->x), mv_round(feat->fwd_match->y));//ͼ1�е������(feat��ƥ���)
            //qDebug()<<"pt2:("<<pt2.x<<","<<pt2.y<<")--->pt1:("<<pt1.x<<","<<pt1.y<<")";//�����Ӧ���

            //ͳ��ƥ��������λ�ù�ϵ�����ж�ͼ1��ͼ2������λ�ù�ϵ
            if (pt2.x > pt1.x)
                invertNum++;

            pt2.x += img1->width;//��������ͼ���������еģ�pt2�ĺ��������ͼ1�Ŀ�ȣ���Ϊ���ߵ��յ�
            //mv_line(stacked_ransac, pt1, pt2, MV_RGB(255, 0, 255), 1, 8, 0);//��ƥ��ͼ�ϻ�������
        }

        //����ͼ1�а�Χƥ���ľ���
        //cvRectangle(stacked_ransac,cvPoint(img1LeftBound,0),cvPoint(img1RightBound,img1->height),MV_RGB(0,255,0),2);
        //����ͼ2�а�Χƥ���ľ���
        //cvRectangle(stacked_ransac,cvPoint(img1->width+img2LeftBound,0),cvPoint(img1->width+img2RightBound,img2->height),MV_RGB(0,0,255),2);

        //mv_named_window(IMG_MATCH2);//��������
        //mv_show_image(IMG_MATCH2, stacked_ransac);//��ʾ��RANSAC�㷨ɸѡ���ƥ��ͼ
        //mv_wait_key(0);


        //����ƥ��ͼ
        //QString name_match_RANSAC = name1;//�ļ�����ԭ�ļ���ȥ����ź��"_match_RANSAC"
        //cvSaveImage(name_match_RANSAC.replace(name_match_RANSAC.lastIndexOf(".", -1) - 1, 1, "_match_RANSAC").toAscii().data(), stacked_ransac);


        /*�����м�����ı任����H������img2�еĵ�任Ϊimg1�еĵ㣬���������img1Ӧ������ͼ��img2Ӧ������ͼ��
        ��ʱimg2�еĵ�pt2��img1�еĶ�Ӧ��pt1��x����Ĺ�ϵ�������ǣ�pt2.x < pt1.x
        ���û��򿪵�img1����ͼ��img2����ͼ����img2�еĵ�pt2��img1�еĶ�Ӧ��pt1��x����Ĺ�ϵ�������ǣ�pt2.x > pt1.x
        ����ͨ��ͳ�ƶ�Ӧ��任ǰ��x�����С��ϵ������֪��img1�ǲ�����ͼ��
        ���img1����ͼ����img1�е�ƥ��㾭H������H_IVT�任��ɵõ�img2�е�ƥ���*/

        //��pt2.x > pt1.x�ĵ�ĸ��������ڵ������80%�����϶�img1������ͼ
        if (invertNum > n_inliers * 0.8)
        {
            H = H.inverse();
            mv_image_t * temp = img2;
            img2 = img1;
            img1 = temp;           
        }


        //���ܳɹ�������任���󣬼�����ͼ���й�ͬ���򣬲ſ��Խ���ȫ��ƴ��

        img2 = load_image(imgfile2);
        if (img2 == NULL) return -1;
        show_image(img2);

        //ƴ��ͼ��img1����ͼ��img2����ͼ
        calc_corner(H, img2);//����ͼ2���ĸ��Ǿ��任�������
        //Ϊƴ�ӽ��ͼxformed����ռ�,�߶�Ϊͼ1ͼ2�߶ȵĽ�С�ߣ�����ͼ2���ϽǺ����½Ǳ任��ĵ��λ�þ���ƴ��ͼ�Ŀ��
        xformed = mv_create_image(mv_size_t(MIN(m_right_top.x, m_right_bottom.x), MIN(img1->height, img2->height)), IPL_DEPTH_8U, 3);
        //�ñ任����H����ͼimg2��ͶӰ�任(�任�������������)������ŵ�xformed��
        mv_perspective(img2, xformed, H);

        //cvWarpPerspective()
        show_image(xformed);                


        ////����ƴ�ӷ���ֱ�ӽ�����ͼimg1���ӵ�xformed�����
        mv_image_t* xformed_simple = mv_clone_image(xformed);//����ƴ��ͼ��������xformed
        //mv_set_image_roi(xformed_simple, mv_rect_t(0, 0, img1->width, img1->height));
        mv_add_weighted(img1, 1, xformed_simple, 0, 0, xformed_simple);
        show_image(xformed_simple);
        //mv_reset_image_roi(xformed_simple);

        


        //////������ƴ��ͼ����¡��xformed
        //mv_image_t* xformed_proc = mv_clone_image(xformed);

        ////�ص�������ߵĲ�����ȫȡ��ͼ1
        //mv_set_image_roi(img1, mv_rect_t(0, 0, MIN(m_left_top.x, m_left_top.x), xformed_proc->height));
        //mv_set_image_roi(xformed, mv_rect_t(0, 0, MIN(m_left_top.x, m_left_top.x), xformed_proc->height));
        //mv_set_image_roi(xformed_proc, mv_rect_t(0, 0, MIN(m_left_top.x, m_left_top.x), xformed_proc->height));
        //mv_add_weighted(img1, 1, xformed, 0, 0, xformed_proc);
        //mv_reset_image_roi(img1);
        //mv_reset_image_roi(xformed);
        //mv_reset_image_roi(xformed_proc);

        //show_image(xformed_proc);
        //

        ////���ü�Ȩƽ���ķ����ں��ص�����
        //int start = MIN(m_left_top.x, m_left_bottom.x);//��ʼλ�ã����ص��������߽�
        //double processWidth = img1->width - start;//�ص�����Ŀ��
        //double alpha = 1;//img1�����ص�Ȩ��
        //for (int i = 0; i < xformed_proc->height; i++)//������
        //{
        //    const unsigned char * pixel_img1 = ((unsigned char *)(img1->imageData + img1->widthStep * i));//img1�е�i�����ݵ�ָ��
        //    const unsigned char * pixel_xformed = ((unsigned char *)(xformed->imageData + xformed->widthStep * i));//xformed�е�i�����ݵ�ָ��
        //    unsigned char * pixel_xformed_proc = ((unsigned char *)(xformed_proc->imageData + xformed_proc->widthStep * i));//xformed_proc�е�i�����ݵ�ָ��
        //    for (int j = start; j < img1->width; j++)//�����ص��������
        //    {
        //        //�������ͼ��xformed�������صĺڵ㣬����ȫ����ͼ1�е�����
        //        if (pixel_xformed[j * 3] < 50 && pixel_xformed[j * 3 + 1] < 50 && pixel_xformed[j * 3 + 2] < 50)
        //        {
        //            alpha = 1;
        //        }
        //        else
        //        {   //img1�����ص�Ȩ�أ��뵱ǰ�������ص�������߽�ľ��������
        //            alpha = (processWidth - (j - start)) / processWidth;
        //        }
        //        pixel_xformed_proc[j * 3] = pixel_img1[j * 3] * alpha + pixel_xformed[j * 3] * (1 - alpha);//Bͨ��
        //        pixel_xformed_proc[j * 3 + 1] = pixel_img1[j * 3 + 1] * alpha + pixel_xformed[j * 3 + 1] * (1 - alpha);//Gͨ��
        //        pixel_xformed_proc[j * 3 + 2] = pixel_img1[j * 3 + 2] * alpha + pixel_xformed[j * 3 + 2] * (1 - alpha);//Rͨ��
        //    }
        //}

        //show_image(xformed_proc);



        ////*�ص�����ȡ����ͼ���ƽ��ֵ��Ч������
        ////����ROI���ǰ����ص�����ľ���
        //mv_set_image_roi(xformed_proc, mv_rect_t(MIN(m_left_top.x, m_left_top.x), 0, img1->width - MIN(leftTop.x, m_left_top.x), xformed_proc->height));
        //mv_set_image_roi(img1, mv_rect_t(MIN(leftTop.x, m_left_top.x), 0, img1->width - MIN(leftTop.x, m_left_top.x), xformed_proc->height));
        //mv_set_image_roi(xformed, mv_rect_t(MIN(leftTop.x, m_left_top.x), 0, img1->width - MIN(leftTop.x, m_left_top.x), xformed_proc->height));
        //mv_add_weighted(img1, 0.5, xformed, 0.5, 0, xformed_proc);
        //mv_reset_image_roi(xformed_proc);
        //mv_reset_image_roi(img1);
        //mv_reset_image_roi(xformed); 


        //show_image(xformed_proc);

        return 0;
    }




private:
    mv_point_t m_left_top;
    mv_point_t m_left_bottom;
    mv_point_t m_right_top;
    mv_point_t m_right_bottom;;
};




int test_image_proc() {

    const char* imgfile1 = "g1.jpg";

    IplImage* img_org = cvLoadImage(imgfile1, 1);
    mv_image_t* img = mv_image_cv2mv(img_org);
    show_image(img);

    mv_image_t* gray8 = mv_create_image(mv_get_size(img), IPL_DEPTH_8U, 1);
    mv_convert_gray(img, gray8);
    show_image(gray8);

    //mv_image_t* gray8_clone = mv_clone_image(gray8);
    //show_image(gray8_clone);


    //show_image(gray32);

    mv_image_t* gray32 = mv_create_image(mv_get_size(img), IPL_DEPTH_32F, 1);
    mv_normalize_u8(gray8, gray32, 1.0 / 255.0);
    //show_image(gray32);

    //mv_image_t* resize1 = mv_create_image(mv_size_t(img->width * 1.2, img->height * 1.2), IPL_DEPTH_8U, 1);
    //mv_resize_nn(gray8_clone, resize1);
    //show_image(resize1);

    mv_image_t* resize2 = mv_create_image(mv_size_t(img->width * 1.3, img->height * 1.3), IPL_DEPTH_32F, 1);
    mv_resize_cubic(gray32, resize2);
    show_image(resize2);

    double sigma = 10;
    mv_image_t* blur = mv_create_image(mv_size_t(img->width, img->height), IPL_DEPTH_32F, 1);
    mv_box_blur(gray32, blur, sigma);
    show_image(blur);

    IplImage* org_gray = cvCreateImage(cvSize(img_org->width, img_org->height), IPL_DEPTH_8U, 1);
    cvCvtColor(img_org, org_gray, CV_BGR2GRAY);
    show_cv_image(org_gray);

    IplImage* blur2 = cvCreateImage(cvSize(img_org->width, img_org->height), IPL_DEPTH_8U, 1);
    cvSmooth(org_gray, blur2, CV_GAUSSIAN, 0, 0, sigma, sigma);
    show_cv_image(blur2);

    mv_image_t* blur_sub = mv_create_image(mv_get_size(img), IPL_DEPTH_8U, 1);
    mv_sub(blur, gray8, blur_sub);
    show_image(blur_sub);

    IplImage* blur2_sub = cvCreateImage(cvSize(img_org->width, img_org->height), IPL_DEPTH_8U, 1);
    cvSub(blur2, org_gray, blur2_sub, NULL);
    show_cv_image(blur2_sub);


    //compare_image(blur_sub, blur2_sub);

    return 0;
}

//
//
//extern "C" {
//#include "meschach/matrix.h"
//#include "meschach/matrix2.h"
//}


void test_matrix()
{
    //MAT* new_m = m_get(2, 2);
    //
    //m_set_val(new_m, 0, 0, 1);
    //m_set_val(new_m, 0, 1, 2);
    //m_set_val(new_m, 1, 0, 3);
    //m_set_val(new_m, 1, 1, 4);

    //MAT* result = m_get(2, 2);
    //m_inverse(new_m, result);

    //VEC* out_v = v_get(2);
    //
    //MAT* new_n = m_get(2, 2);
    //m_set_val(new_n, 0, 0, 4);
    //m_set_val(new_n, 0, 1, 3);
    //m_set_val(new_n, 1, 0, 2);
    //m_set_val(new_n, 1, 1, 1);
    //svd(new_m, NULL, new_n, out_v);
    //WRITE_INFO_LOG("%f, %f", out_v->ve[0], out_v->ve[1]);
    //
    //CvMat* H = cvCreateMat(2, 2, CV_64FC1);
    //cvmSet(H, 0, 0, 1);
    //cvmSet(H, 0, 1, 2);
    //cvmSet(H, 1, 0, 3);
    //cvmSet(H, 1, 1, 4);
    //

    //CvMat* V = cvCreateMat(2, 2, CV_64FC1);
    //cvmSet(V, 0, 0, 4);
    //cvmSet(V, 0, 1, 3);
    //cvmSet(V, 1, 0, 2);
    //cvmSet(V, 1, 1, 1);

    //CvMat* R = cvCreateMat(2, 2, CV_64FC1);
    //cvSolve(H, V, R, CV_SVD);
    //WRITE_INFO_LOG("%f, %f", cvmGet(R, 0, 0), cvmGet(R, 0, 1));
    //WRITE_INFO_LOG("%f, %f", cvmGet(R, 1, 0), cvmGet(R, 1, 1));
    //
    //WRITE_INFO_LOG("s");

    //m_add()	        Add matrices
    //m_mlt()	        Multiplies matrices
    //m_sub()		    Subtract matrices
    //mv_mlt()	        Computes  Ax
    //mv_mltadd()	    Computes  y <-Ax + y
    //m_zero()	        Zero a matrix
    //cvInvert()
}


int main(int argc, char** argv)
{
    image_stitching* instance = new image_stitching();
    instance->image_stitching_entry();
    //image_stitching_entry();
    //test_image_proc();
    //test_matrix();
    //cvWaitKey(0);
    return 0;
}




