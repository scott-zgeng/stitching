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

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include <stdio.h>


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



CvPoint leftTop, leftBottom, rightTop, rightBottom;


//����ͼ2���ĸ��Ǿ�����H�任�������
void CalcFourCorner(CvMat * H, IplImage* img2)
{
    //����ͼ2���ĸ��Ǿ�����H�任�������
    double v2[] = { 0, 0, 1 };//���Ͻ�
    double v1[3];//�任�������ֵ
    CvMat V2 = cvMat(3, 1, CV_64FC1, v2);
    CvMat V1 = cvMat(3, 1, CV_64FC1, v1);
    cvGEMM(H, &V2, 1, 0, 1, &V1);//����˷�
    leftTop.x = cvRound(v1[0] / v1[2]);
    leftTop.y = cvRound(v1[1] / v1[2]);
    //cvCircle(xformed,leftTop,7,CV_RGB(255,0,0),2);

    //��v2��������Ϊ���½�����
    v2[0] = 0;
    v2[1] = img2->height;
    V2 = cvMat(3, 1, CV_64FC1, v2);
    V1 = cvMat(3, 1, CV_64FC1, v1);
    cvGEMM(H, &V2, 1, 0, 1, &V1);
    leftBottom.x = cvRound(v1[0] / v1[2]);
    leftBottom.y = cvRound(v1[1] / v1[2]);
    //cvCircle(xformed,leftBottom,7,CV_RGB(255,0,0),2);

    //��v2��������Ϊ���Ͻ�����
    v2[0] = img2->width;
    v2[1] = 0;
    V2 = cvMat(3, 1, CV_64FC1, v2);
    V1 = cvMat(3, 1, CV_64FC1, v1);
    cvGEMM(H, &V2, 1, 0, 1, &V1);
    rightTop.x = cvRound(v1[0] / v1[2]);
    rightTop.y = cvRound(v1[1] / v1[2]);
    //cvCircle(xformed,rightTop,7,CV_RGB(255,0,0),2);

    //��v2��������Ϊ���½�����
    v2[0] = img2->width;
    v2[1] = img2->height;
    V2 = cvMat(3, 1, CV_64FC1, v2);
    V1 = cvMat(3, 1, CV_64FC1, v1);
    cvGEMM(H, &V2, 1, 0, 1, &V1);
    rightBottom.x = cvRound(v1[0] / v1[2]);
    rightBottom.y = cvRound(v1[1] / v1[2]);
    //cvCircle(xformed,rightBottom,7,CV_RGB(255,0,0),2);

}

int main(int argc, char** argv)
{
    WRITE_INFO_LOG("enter main function");

    IplImage* img1, *img2, *stacked;
    struct feature* feat1, *feat2, *feat;
    struct feature** nbrs;
    struct kd_node* kd_root;
    CvPoint pt1, pt2;
    double d0, d1;
    int n1, n2, k, i, m = 0;


    const char* imgfile1 = "g1.jpg";
    const char* imgfile2 = "g2.jpg";


    img1 = cvLoadImage(imgfile1, 1);
    if (!img1) {
        WRITE_ERROR_LOG("unable to load image from %s", imgfile1);
        return 1;
    }

    img2 = cvLoadImage(imgfile2, 1);
    if (!img2) {
        WRITE_ERROR_LOG("unable to load image from %s", imgfile2);
        return 1;
    }


    stacked = stack_imgs(img1, img2);

    WRITE_INFO_LOG("Finding features in %s...", imgfile1);
    n1 = sift_features(img1, &feat1);

    WRITE_INFO_LOG("Finding features in %s...", imgfile2);
    n2 = sift_features(img2, &feat2);

    WRITE_INFO_LOG("Building kd tree...");
    kd_root = kdtree_build(feat1, n1);

    for (i = 0; i < n2; i++)
    {
        feat = feat2 + i;
        k = kdtree_bbf_knn(kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS);
        if (k == 2)
        {
            d0 = descr_dist_sq(feat, nbrs[0]);
            d1 = descr_dist_sq(feat, nbrs[1]);
            if (d0 < d1 * NN_SQ_DIST_RATIO_THR)
            {
                pt1 = cvPoint(cvRound(feat->x), cvRound(feat->y));
                pt2 = cvPoint(cvRound(nbrs[0]->x), cvRound(nbrs[0]->y));
                pt2.y += img1->height;
                cvLine(stacked, pt1, pt2, CV_RGB(255, 0, 255), 1, 8, 0);
                m++;
                feat2[i].fwd_match = nbrs[0];
            }
        }
        free(nbrs);
    }

    WRITE_INFO_LOG("Found %d total matches", m);
    display_big_img(stacked, "Matches");
    cvWaitKey(0);

    /*
       UNCOMMENT BELOW TO SEE HOW RANSAC FUNCTION WORKS
       Note that this line above:
       feat1[i].fwd_match = nbrs[0];
       is important for the RANSAC function to work.
       */


    CvMat* H;
    IplImage* xformed;
    struct feature **inliers;
    int n_inliers;

    H = ransac_xform(feat2, n2, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01, homog_xfer_err, 3.0, &inliers, &n_inliers);
    IplImage *stacked_ransac;

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
            pt2 = CvPoint(cvRound(feat->x), cvRound(feat->y));//ͼ2�е������
            pt1 = CvPoint(cvRound(feat->fwd_match->x), cvRound(feat->fwd_match->y));//ͼ1�е������(feat��ƥ���)
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
            cvLine(stacked_ransac, pt1, pt2, CV_RGB(255, 0, 255), 1, 8, 0);//��ƥ��ͼ�ϻ�������
        }

        //����ͼ1�а�Χƥ���ľ���
        //cvRectangle(stacked_ransac,cvPoint(img1LeftBound,0),cvPoint(img1RightBound,img1->height),CV_RGB(0,255,0),2);
        //����ͼ2�а�Χƥ���ľ���
        //cvRectangle(stacked_ransac,cvPoint(img1->width+img2LeftBound,0),cvPoint(img1->width+img2RightBound,img2->height),CV_RGB(0,0,255),2);

        cvNamedWindow(IMG_MATCH2);//��������
        cvShowImage(IMG_MATCH2, stacked_ransac);//��ʾ��RANSAC�㷨ɸѡ���ƥ��ͼ
        cvWaitKey(0);


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
            
            CvMat * H_IVT = cvCreateMat(3, 3, CV_64FC1);//�任����������
            //��H������H_IVTʱ�����ɹ���������ط���ֵ
            if (cvInvert(H, H_IVT))
            {
                
                cvReleaseMat(&H);//�ͷű任����H����Ϊ�ò�����
                H = cvCloneMat(H_IVT);//��H������H_IVT�е����ݿ�����H��
                cvReleaseMat(&H_IVT);//�ͷ�����H_IVT
                //��img1��img2�Ե�
                IplImage * temp = img2;
                img2 = img1;
                img1 = temp;
                //cvShowImage(IMG1,img1);
                //cvShowImage(IMG2,img2);
                //ui->mosaicButton->setEnabled(true);//����ȫ��ƴ�Ӱ�ť
            }
            else//H������ʱ������0
            {
                cvReleaseMat(&H_IVT);//�ͷ�����H_IVT
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
        xformed = cvCreateImage(cvSize(MIN(rightTop.x, rightBottom.x), MIN(img1->height, img2->height)), IPL_DEPTH_8U, 3);
        //�ñ任����H����ͼimg2��ͶӰ�任(�任�������������)������ŵ�xformed��
        cvWarpPerspective(img2, xformed, H, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
        cvNamedWindow(IMG_MOSAIC_TEMP); //��ʾ��ʱͼ,��ֻ��ͼ2�任���ͼ
        cvShowImage(IMG_MOSAIC_TEMP, xformed);
        cvWaitKey(0);


        //����ƴ�ӷ���ֱ�ӽ�����ͼimg1���ӵ�xformed�����
        IplImage* xformed_simple = cvCloneImage(xformed);//����ƴ��ͼ��������xformed
        cvSetImageROI(xformed_simple, cvRect(0, 0, img1->width, img1->height));
        cvAddWeighted(img1, 1, xformed_simple, 0, 0, xformed_simple);
        cvResetImageROI(xformed_simple);
        cvNamedWindow(IMG_MOSAIC_SIMPLE);//��������
        cvShowImage(IMG_MOSAIC_SIMPLE, xformed_simple);//��ʾ����ƴ��ͼ
        cvWaitKey(0);


        //������ƴ��ͼ����¡��xformed
        IplImage* xformed_proc = cvCloneImage(xformed);

        //�ص�������ߵĲ�����ȫȡ��ͼ1
        cvSetImageROI(img1, cvRect(0, 0, MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        cvSetImageROI(xformed, cvRect(0, 0, MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        cvSetImageROI(xformed_proc, cvRect(0, 0, MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        cvAddWeighted(img1, 1, xformed, 0, 0, xformed_proc);
        cvResetImageROI(img1);
        cvResetImageROI(xformed);
        cvResetImageROI(xformed_proc);
        cvNamedWindow(IMG_MOSAIC_BEFORE_FUSION);
        cvShowImage(IMG_MOSAIC_BEFORE_FUSION, xformed_proc);//��ʾ�ں�֮ǰ��ƴ��ͼ
        cvWaitKey(0);

        //���ü�Ȩƽ���ķ����ں��ص�����
        int start = MIN(leftTop.x, leftBottom.x);//��ʼλ�ã����ص��������߽�
        double processWidth = img1->width - start;//�ص�����Ŀ��
        double alpha = 1;//img1�����ص�Ȩ��
        for (int i = 0; i<xformed_proc->height; i++)//������
        {
            const uchar * pixel_img1 = ((uchar *)(img1->imageData + img1->widthStep * i));//img1�е�i�����ݵ�ָ��
            const uchar * pixel_xformed = ((uchar *)(xformed->imageData + xformed->widthStep * i));//xformed�е�i�����ݵ�ָ��
            uchar * pixel_xformed_proc = ((uchar *)(xformed_proc->imageData + xformed_proc->widthStep * i));//xformed_proc�е�i�����ݵ�ָ��
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
        cvNamedWindow(IMG_MOSAIC_PROC);//��������
        cvShowImage(IMG_MOSAIC_PROC, xformed_proc);//��ʾ������ƴ��ͼ
        cvWaitKey(0);

        //*�ص�����ȡ����ͼ���ƽ��ֵ��Ч������
        //����ROI���ǰ����ص�����ľ���
        cvSetImageROI(xformed_proc, cvRect(MIN(leftTop.x, leftBottom.x), 0, img1->width - MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        cvSetImageROI(img1, cvRect(MIN(leftTop.x, leftBottom.x), 0, img1->width - MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        cvSetImageROI(xformed, cvRect(MIN(leftTop.x, leftBottom.x), 0, img1->width - MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        cvAddWeighted(img1, 0.5, xformed, 0.5, 0, xformed_proc);
        cvResetImageROI(xformed_proc);
        cvResetImageROI(img1);
        cvResetImageROI(xformed); //*/

        /*��ƴ�ӷ���Χ��������˲�������ƴ�ӷ죬Ч������
        //�ڴ���ǰ���ͼ�Ϸֱ����ú��ƴ�ӷ�ľ���ROI
        cvSetImageROI(xformed_proc,cvRect(img1->width-10,0,img1->width+10,xformed->height));
        cvSetImageROI(xformed,cvRect(img1->width-10,0,img1->width+10,xformed->height));
        cvSmooth(xformed,xformed_proc,CV_MEDIAN,5);//��ƴ�ӷ���Χ���������ֵ�˲�
        cvResetImageROI(xformed);
        cvResetImageROI(xformed_proc);
        cvShowImage(IMG_MOSAIC_PROC,xformed_proc);//��ʾ������ƴ��ͼ */

        /*��ͨ���񻯽���任���ͼ��ʧ������⣬����Ť�������ͼ��Ч������
        double a[]={  0, -1,  0, -1,  5, -1, 0, -1,  0  };//������˹�˲��˵�����
        CvMat kernel = cvMat(3,3,CV_64FC1,a);//������˹�˲���
        cvFilter2D(xformed_proc,xformed_proc,&kernel);//�˲�
        cvShowImage(IMG_MOSAIC_PROC,xformed_proc);//��ʾ������ƴ��ͼ*/
        
    }



    cvReleaseImage(&stacked);
    cvReleaseImage(&img1);
    cvReleaseImage(&img2);
    kdtree_release(kd_root);
    free(feat1);
    free(feat2);
    return 0;
}
