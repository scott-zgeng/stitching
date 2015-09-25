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

#include "mv_base.h"


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



mv_point_t leftTop, leftBottom, rightTop, rightBottom;

void CalcFourCorner(mv_matrix_t * H, mv_image_t* img2);


int main(int argc, char** argv)
{
    WRITE_INFO_LOG("enter main function");

    //mv_image_t* img1, *img2, *stacked;
    struct feature *feat;
    struct feature** nbrs;
    struct kd_node* kd_root;
    mv_point_t pt1, pt2;
    double d0, d1;
    int n1, n2, k, i, m = 0;

    
    //const char* imgfile1 = "img_left.jpg";
    //const char* imgfile2 = "img_right.jpg";
    const char* imgfile1 = "g1.jpg";
    const char* imgfile2 = "g2.jpg";


    mv_image_t* img1 = mv_load_image(imgfile1, 1);
    if (!img1) {
        WRITE_ERROR_LOG("unable to load image from %s", imgfile1);
        return 1;
    }

    mv_image_t* img2 = mv_load_image(imgfile2, 1);
    if (!img2) {
        WRITE_ERROR_LOG("unable to load image from %s", imgfile2);
        return 1;
    }


    //stacked = stack_imgs(img1, img2);

    WRITE_INFO_LOG("Finding features in %s...", imgfile1);
    mv_features feat1;
    n1 = sift_features(img1, &feat1);

    mv_features feat2;
    WRITE_INFO_LOG("Finding features in %s...", imgfile2);
    n2 = sift_features(img2, &feat2);

    

    WRITE_INFO_LOG("Building kd tree...");
    kd_root = kdtree_build(feat1[0], feat1.size());

    for (i = 0; i < n2; i++)
    {
        feat = feat2[i];
        k = kdtree_bbf_knn(kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS);
        if (k == 2)
        {
            d0 = descr_dist_sq(feat, nbrs[0]);
            d1 = descr_dist_sq(feat, nbrs[1]);
            if (d0 < d1 * NN_SQ_DIST_RATIO_THR)
            {
                pt1 = mv_point_t(mv_round(feat->x), mv_round(feat->y));
                pt2 = mv_point_t(mv_round(nbrs[0]->x), mv_round(nbrs[0]->y));
                pt2.y += img1->height;
                //mv_line(stacked, pt1, pt2, MV_RGB(255, 0, 255), 1, 8, 0);
                m++;
                feat2[i]->fwd_match = nbrs[0];
            }
        }
        free(nbrs);
    }

    WRITE_INFO_LOG("Found %d total matches", m);
    //display_big_img(stacked, "Matches");
    //mv_wait_key(0);

    /*
       UNCOMMENT BELOW TO SEE HOW RANSAC FUNCTION WORKS
       Note that this line above:
       feat1[i].fwd_match = nbrs[0];
       is important for the RANSAC function to work.
       */


    mv_matrix_t* H;
    mv_image_t* xformed;
    struct feature **inliers;
    int n_inliers;

    H = ransac_xform(feat2[0], feat2.size(), FEATURE_FWD_MATCH, lsq_homog, 4, 0.01, homog_xfer_err, 3.0, &inliers, &n_inliers);
    mv_image_t *stacked_ransac;

    //���ܳɹ�������任���󣬼�����ͼ���й�ͬ����
    if (H)
    {        
        stacked_ransac = stack_imgs_horizontal(img1, img2);//�ϳ�ͼ����ʾ��RANSAC�㷨ɸѡ���ƥ����

        //img1LeftBound = inliers[0]->fwd_match->x;//ͼ1��ƥ�����Ӿ��ε���߽�
        //img1RightBound = img1LeftBound;//ͼ1��ƥ�����Ӿ��ε��ұ߽�
        //img2LeftBound = inliers[0]->x;//ͼ2��ƥ�����Ӿ��ε���߽�
        //img2RightBound = img2LeftBound;//ͼ2��ƥ�����Ӿ��ε��ұ߽�

        int invertNum = 0;//ͳ��pt2.x > pt1.x��ƥ���Եĸ��������ж�img1���Ƿ���ͼ

        //������RANSAC�㷨ɸѡ��������㼯��inliers���ҵ�ÿ���������ƥ��㣬��������
        for (int i = 0; i<n_inliers; i++)
        {
            feat = inliers[i];//��i��������
            pt2 = mv_point_t(mv_round(feat->x), mv_round(feat->y));//ͼ2�е������
            pt1 = mv_point_t(mv_round(feat->fwd_match->x), mv_round(feat->fwd_match->y));//ͼ1�е������(feat��ƥ���)
            //qDebug()<<"pt2:("<<pt2.x<<","<<pt2.y<<")--->pt1:("<<pt1.x<<","<<pt1.y<<")";//�����Ӧ���

            /*��ƥ�������ı߽�
            if(pt1.x < img1LeftBound) img1LeftBound = pt1.x;
            if(pt1.x > img1RightBound) img1RightBound = pt1.x;
            if(pt2.x < img2LeftBound) img2LeftBound = pt2.x;
            if(pt2.x > img2RightBound) img2RightBound = pt2.x;//*/

            //ͳ��ƥ��������λ�ù�ϵ�����ж�ͼ1��ͼ2������λ�ù�ϵ
            if (pt2.x > pt1.x)
                invertNum++;
            
            pt2.x += img1->width;//��������ͼ���������еģ�pt2�ĺ��������ͼ1�Ŀ�ȣ���Ϊ���ߵ��յ�
            mv_line(stacked_ransac, pt1, pt2, MV_RGB(255, 0, 255), 1, 8, 0);//��ƥ��ͼ�ϻ�������
        }

        //����ͼ1�а�Χƥ���ľ���
        //cvRectangle(stacked_ransac,cvPoint(img1LeftBound,0),cvPoint(img1RightBound,img1->height),MV_RGB(0,255,0),2);
        //����ͼ2�а�Χƥ���ľ���
        //cvRectangle(stacked_ransac,cvPoint(img1->width+img2LeftBound,0),cvPoint(img1->width+img2RightBound,img2->height),MV_RGB(0,0,255),2);

        mv_named_window(IMG_MATCH2);//��������
        mv_show_image(IMG_MATCH2, stacked_ransac);//��ʾ��RANSAC�㷨ɸѡ���ƥ��ͼ
        mv_wait_key(0);


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
            
            mv_matrix_t * H_IVT = mv_create_matrix(3, 3, MV_64FC1);//�任����������
            //��H������H_IVTʱ�����ɹ���������ط���ֵ
            if (mv_invert(H, H_IVT, MV_LU))
            {
                
                mv_release_matrix(&H);//�ͷű任����H����Ϊ�ò�����
                H = mv_clone_matrix(H_IVT);//��H������H_IVT�е����ݿ�����H��
                mv_release_matrix(&H_IVT);//�ͷ�����H_IVT
                //��img1��img2�Ե�
                mv_image_t * temp = img2;
                img2 = img1;
                img1 = temp;
                //cvShowImage(IMG1,img1);
                //cvShowImage(IMG2,img2);
                //ui->mosaicButton->setEnabled(true);//����ȫ��ƴ�Ӱ�ť
            }
            else//H������ʱ������0
            {
                mv_release_matrix(&H_IVT);//�ͷ�����H_IVT
                //QMessageBox::warning(this, tr("����"), tr("�任����H������"));
            }
        }        
    }


    //���ܳɹ�������任���󣬼�����ͼ���й�ͬ���򣬲ſ��Խ���ȫ��ƴ��
    if (H)
    {
        //ƴ��ͼ��img1����ͼ��img2����ͼ
        CalcFourCorner(H, img2);//����ͼ2���ĸ��Ǿ��任�������
        //Ϊƴ�ӽ��ͼxformed����ռ�,�߶�Ϊͼ1ͼ2�߶ȵĽ�С�ߣ�����ͼ2���ϽǺ����½Ǳ任��ĵ��λ�þ���ƴ��ͼ�Ŀ��
        xformed = mv_create_image(mv_size_t(MIN(rightTop.x, rightBottom.x), MIN(img1->height, img2->height)), IPL_DEPTH_8U, 3);
        //�ñ任����H����ͼimg2��ͶӰ�任(�任�������������)������ŵ�xformed��
        mv_warp_perspective(img2, xformed, H, MV_INTER_LINEAR + MV_WARP_FILL_OUTLIERS, mv_scalar_t(0));
        mv_named_window(IMG_MOSAIC_TEMP); //��ʾ��ʱͼ,��ֻ��ͼ2�任���ͼ
        mv_show_image(IMG_MOSAIC_TEMP, xformed);
        mv_wait_key(0);


        //����ƴ�ӷ���ֱ�ӽ�����ͼimg1���ӵ�xformed�����
        mv_image_t* xformed_simple = mv_clone_image(xformed);//����ƴ��ͼ��������xformed
        mv_set_image_roi(xformed_simple, mv_rect_t(0, 0, img1->width, img1->height));
        mv_add_weighted(img1, 1, xformed_simple, 0, 0, xformed_simple);
        mv_reset_image_roi(xformed_simple);
        mv_named_window(IMG_MOSAIC_SIMPLE);//��������
        mv_show_image(IMG_MOSAIC_SIMPLE, xformed_simple);//��ʾ����ƴ��ͼ
        mv_wait_key(0);


        //������ƴ��ͼ����¡��xformed
        mv_image_t* xformed_proc = mv_clone_image(xformed);

        //�ص�������ߵĲ�����ȫȡ��ͼ1
        mv_set_image_roi(img1, mv_rect_t(0, 0, MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        mv_set_image_roi(xformed, mv_rect_t(0, 0, MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        mv_set_image_roi(xformed_proc, mv_rect_t(0, 0, MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        mv_add_weighted(img1, 1, xformed, 0, 0, xformed_proc);
        mv_reset_image_roi(img1);
        mv_reset_image_roi(xformed);
        mv_reset_image_roi(xformed_proc);
        mv_named_window(IMG_MOSAIC_BEFORE_FUSION);
        mv_show_image(IMG_MOSAIC_BEFORE_FUSION, xformed_proc);//��ʾ�ں�֮ǰ��ƴ��ͼ
        mv_wait_key(0);

        //���ü�Ȩƽ���ķ����ں��ص�����
        int start = MIN(leftTop.x, leftBottom.x);//��ʼλ�ã����ص��������߽�
        double processWidth = img1->width - start;//�ص�����Ŀ��
        double alpha = 1;//img1�����ص�Ȩ��
        for (int i = 0; i<xformed_proc->height; i++)//������
        {
            const unsigned char * pixel_img1 = ((unsigned char *)(img1->imageData + img1->widthStep * i));//img1�е�i�����ݵ�ָ��
            const unsigned char * pixel_xformed = ((unsigned char *)(xformed->imageData + xformed->widthStep * i));//xformed�е�i�����ݵ�ָ��
            unsigned char * pixel_xformed_proc = ((unsigned char *)(xformed_proc->imageData + xformed_proc->widthStep * i));//xformed_proc�е�i�����ݵ�ָ��
            for (int j = start; j<img1->width; j++)//�����ص��������
            {
                //�������ͼ��xformed�������صĺڵ㣬����ȫ����ͼ1�е�����
                if (pixel_xformed[j * 3] < 50 && pixel_xformed[j * 3 + 1] < 50 && pixel_xformed[j * 3 + 2] < 50)
                {
                    alpha = 1;
                }
                else
                {   //img1�����ص�Ȩ�أ��뵱ǰ�������ص�������߽�ľ��������
                    alpha = (processWidth - (j - start)) / processWidth;
                }
                pixel_xformed_proc[j * 3] = pixel_img1[j * 3] * alpha + pixel_xformed[j * 3] * (1 - alpha);//Bͨ��
                pixel_xformed_proc[j * 3 + 1] = pixel_img1[j * 3 + 1] * alpha + pixel_xformed[j * 3 + 1] * (1 - alpha);//Gͨ��
                pixel_xformed_proc[j * 3 + 2] = pixel_img1[j * 3 + 2] * alpha + pixel_xformed[j * 3 + 2] * (1 - alpha);//Rͨ��
            }
        }
        mv_named_window(IMG_MOSAIC_PROC);//��������
        mv_show_image(IMG_MOSAIC_PROC, xformed_proc);//��ʾ������ƴ��ͼ
        mv_wait_key(0);

        //*�ص�����ȡ����ͼ���ƽ��ֵ��Ч������
        //����ROI���ǰ����ص�����ľ���
        mv_set_image_roi(xformed_proc, mv_rect_t(MIN(leftTop.x, leftBottom.x), 0, img1->width - MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        mv_set_image_roi(img1, mv_rect_t(MIN(leftTop.x, leftBottom.x), 0, img1->width - MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        mv_set_image_roi(xformed, mv_rect_t(MIN(leftTop.x, leftBottom.x), 0, img1->width - MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        mv_add_weighted(img1, 0.5, xformed, 0.5, 0, xformed_proc);
        mv_reset_image_roi(xformed_proc);
        mv_reset_image_roi(img1);
        mv_reset_image_roi(xformed); //*/

        /*��ƴ�ӷ���Χ��������˲�������ƴ�ӷ죬Ч������
        //�ڴ���ǰ���ͼ�Ϸֱ����ú��ƴ�ӷ�ľ���ROI
        cvSetImageROI(xformed_proc,cvRect(img1->width-10,0,img1->width+10,xformed->height));
        cvSetImageROI(xformed,cvRect(img1->width-10,0,img1->width+10,xformed->height));
        cvSmooth(xformed,xformed_proc,MV_MEDIAN,5);//��ƴ�ӷ���Χ���������ֵ�˲�
        cvResetImageROI(xformed);
        cvResetImageROI(xformed_proc);
        cvShowImage(IMG_MOSAIC_PROC,xformed_proc);//��ʾ������ƴ��ͼ */

        /*��ͨ���񻯽���任���ͼ��ʧ������⣬����Ť�������ͼ��Ч������
        double a[]={  0, -1,  0, -1,  5, -1, 0, -1,  0  };//������˹�˲��˵�����
        CvMat kernel = cvMat(3,3,MV_64FC1,a);//������˹�˲���
        cvFilter2D(xformed_proc,xformed_proc,&kernel);//�˲�
        cvShowImage(IMG_MOSAIC_PROC,xformed_proc);//��ʾ������ƴ��ͼ*/
        
    }


    //mv_release_image(&stacked);
    mv_release_image(&img1);
    mv_release_image(&img2);
    kdtree_release(kd_root);
    //free(feat1);
    //free(feat2);
    return 0;
}



//����ͼ2���ĸ��Ǿ�����H�任�������
void CalcFourCorner(mv_matrix_t * H, mv_image_t* img2)
{
    //����ͼ2���ĸ��Ǿ�����H�任�������
    double v2[] = { 0, 0, 1 };//���Ͻ�
    double v1[3];//�任�������ֵ
    mv_matrix_t V2(3, 1, MV_64FC1, v2);
    mv_matrix_t V1(3, 1, MV_64FC1, v1);
    mv_matrix_mul_add_ex(H, &V2, 1, 0, 1, &V1, 0);//����˷�
    leftTop.x = mv_round(v1[0] / v1[2]);
    leftTop.y = mv_round(v1[1] / v1[2]);
    //cvCircle(xformed,leftTop,7,MV_RGB(255,0,0),2);

    //��v2��������Ϊ���½�����
    v2[0] = 0;
    v2[1] = img2->height;
    V2 = mv_matrix_t(3, 1, MV_64FC1, v2);
    V1 = mv_matrix_t(3, 1, MV_64FC1, v1);
    mv_matrix_mul_add_ex(H, &V2, 1, 0, 1, &V1, 0);
    leftBottom.x = mv_round(v1[0] / v1[2]);
    leftBottom.y = mv_round(v1[1] / v1[2]);
    //cvCircle(xformed,leftBottom,7,MV_RGB(255,0,0),2);

    //��v2��������Ϊ���Ͻ�����
    v2[0] = img2->width;
    v2[1] = 0;
    V2 = mv_matrix_t(3, 1, MV_64FC1, v2);
    V1 = mv_matrix_t(3, 1, MV_64FC1, v1);
    mv_matrix_mul_add_ex(H, &V2, 1, 0, 1, &V1, 0);
    rightTop.x = mv_round(v1[0] / v1[2]);
    rightTop.y = mv_round(v1[1] / v1[2]);
    //cvCircle(xformed,rightTop,7,MV_RGB(255,0,0),2);

    //��v2��������Ϊ���½�����
    v2[0] = img2->width;
    v2[1] = img2->height;
    V2 = mv_matrix_t(3, 1, MV_64FC1, v2);
    V1 = mv_matrix_t(3, 1, MV_64FC1, v1);
    mv_matrix_mul_add_ex(H, &V2, 1, 0, 1, &V1, 0);
    rightBottom.x = mv_round(v1[0] / v1[2]);
    rightBottom.y = mv_round(v1[1] / v1[2]);
    //cvCircle(xformed,rightBottom,7,MV_RGB(255,0,0),2);

}