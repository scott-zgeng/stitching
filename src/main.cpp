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


//窗口名字符串
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


//计算图2的四个角经矩阵H变换后的坐标
void CalcFourCorner(CvMat * H, IplImage* img2)
{
    //计算图2的四个角经矩阵H变换后的坐标
    double v2[] = { 0, 0, 1 };//左上角
    double v1[3];//变换后的坐标值
    CvMat V2 = cvMat(3, 1, CV_64FC1, v2);
    CvMat V1 = cvMat(3, 1, CV_64FC1, v1);
    cvGEMM(H, &V2, 1, 0, 1, &V1);//矩阵乘法
    leftTop.x = cvRound(v1[0] / v1[2]);
    leftTop.y = cvRound(v1[1] / v1[2]);
    //cvCircle(xformed,leftTop,7,CV_RGB(255,0,0),2);

    //将v2中数据设为左下角坐标
    v2[0] = 0;
    v2[1] = img2->height;
    V2 = cvMat(3, 1, CV_64FC1, v2);
    V1 = cvMat(3, 1, CV_64FC1, v1);
    cvGEMM(H, &V2, 1, 0, 1, &V1);
    leftBottom.x = cvRound(v1[0] / v1[2]);
    leftBottom.y = cvRound(v1[1] / v1[2]);
    //cvCircle(xformed,leftBottom,7,CV_RGB(255,0,0),2);

    //将v2中数据设为右上角坐标
    v2[0] = img2->width;
    v2[1] = 0;
    V2 = cvMat(3, 1, CV_64FC1, v2);
    V1 = cvMat(3, 1, CV_64FC1, v1);
    cvGEMM(H, &V2, 1, 0, 1, &V1);
    rightTop.x = cvRound(v1[0] / v1[2]);
    rightTop.y = cvRound(v1[1] / v1[2]);
    //cvCircle(xformed,rightTop,7,CV_RGB(255,0,0),2);

    //将v2中数据设为右下角坐标
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

    //若能成功计算出变换矩阵，即两幅图中有共同区域
    if (H)
    {        
        stacked_ransac = stack_imgs_horizontal(img1, img2);//合成图像，显示经RANSAC算法筛选后的匹配结果

        //img1LeftBound = inliers[0]->fwd_match->x;//图1中匹配点外接矩形的左边界
        //img1RightBound = img1LeftBound;//图1中匹配点外接矩形的右边界
        //img2LeftBound = inliers[0]->x;//图2中匹配点外接矩形的左边界
        //img2RightBound = img2LeftBound;//图2中匹配点外接矩形的右边界

        int invertNum = 0;//统计pt2.x > pt1.x的匹配点对的个数，来判断img1中是否右图

        //遍历经RANSAC算法筛选后的特征点集合inliers，找到每个特征点的匹配点，画出连线
        for (int i = 0; i<n_inliers; i++)
        {
            feat = inliers[i];//第i个特征点
            pt2 = CvPoint(cvRound(feat->x), cvRound(feat->y));//图2中点的坐标
            pt1 = CvPoint(cvRound(feat->fwd_match->x), cvRound(feat->fwd_match->y));//图1中点的坐标(feat的匹配点)
            //qDebug()<<"pt2:("<<pt2.x<<","<<pt2.y<<")--->pt1:("<<pt1.x<<","<<pt1.y<<")";//输出对应点对

            /*找匹配点区域的边界
            if(pt1.x < img1LeftBound) img1LeftBound = pt1.x;
            if(pt1.x > img1RightBound) img1RightBound = pt1.x;
            if(pt2.x < img2LeftBound) img2LeftBound = pt2.x;
            if(pt2.x > img2RightBound) img2RightBound = pt2.x;//*/

            //统计匹配点的左右位置关系，来判断图1和图2的左右位置关系
            if (pt2.x > pt1.x)
                invertNum++;
            
            pt2.x += img1->width;//由于两幅图是左右排列的，pt2的横坐标加上图1的宽度，作为连线的终点
            cvLine(stacked_ransac, pt1, pt2, CV_RGB(255, 0, 255), 1, 8, 0);//在匹配图上画出连线
        }

        //绘制图1中包围匹配点的矩形
        //cvRectangle(stacked_ransac,cvPoint(img1LeftBound,0),cvPoint(img1RightBound,img1->height),CV_RGB(0,255,0),2);
        //绘制图2中包围匹配点的矩形
        //cvRectangle(stacked_ransac,cvPoint(img1->width+img2LeftBound,0),cvPoint(img1->width+img2RightBound,img2->height),CV_RGB(0,0,255),2);

        cvNamedWindow(IMG_MATCH2);//创建窗口
        cvShowImage(IMG_MATCH2, stacked_ransac);//显示经RANSAC算法筛选后的匹配图
        cvWaitKey(0);


        //保存匹配图
        //QString name_match_RANSAC = name1;//文件名，原文件名去掉序号后加"_match_RANSAC"
        //cvSaveImage(name_match_RANSAC.replace(name_match_RANSAC.lastIndexOf(".", -1) - 1, 1, "_match_RANSAC").toAscii().data(), stacked_ransac);


        /*程序中计算出的变换矩阵H用来将img2中的点变换为img1中的点，正常情况下img1应该是左图，img2应该是右图。
        此时img2中的点pt2和img1中的对应点pt1的x坐标的关系基本都是：pt2.x < pt1.x
        若用户打开的img1是右图，img2是左图，则img2中的点pt2和img1中的对应点pt1的x坐标的关系基本都是：pt2.x > pt1.x
        所以通过统计对应点变换前后x坐标大小关系，可以知道img1是不是右图。
        如果img1是右图，将img1中的匹配点经H的逆阵H_IVT变换后可得到img2中的匹配点*/

        //若pt2.x > pt1.x的点的个数大于内点个数的80%，则认定img1中是右图
        if (invertNum > n_inliers * 0.8)
        {
            
            CvMat * H_IVT = cvCreateMat(3, 3, CV_64FC1);//变换矩阵的逆矩阵
            //求H的逆阵H_IVT时，若成功求出，返回非零值
            if (cvInvert(H, H_IVT))
            {
                
                cvReleaseMat(&H);//释放变换矩阵H，因为用不到了
                H = cvCloneMat(H_IVT);//将H的逆阵H_IVT中的数据拷贝到H中
                cvReleaseMat(&H_IVT);//释放逆阵H_IVT
                //将img1和img2对调
                IplImage * temp = img2;
                img2 = img1;
                img1 = temp;
                //cvShowImage(IMG1,img1);
                //cvShowImage(IMG2,img2);
                //ui->mosaicButton->setEnabled(true);//激活全景拼接按钮
            }
            else//H不可逆时，返回0
            {
                cvReleaseMat(&H_IVT);//释放逆阵H_IVT
                //QMessageBox::warning(this, tr("警告"), tr("变换矩阵H不可逆"));
            }
        }        
    }


    //若能成功计算出变换矩阵，即两幅图中有共同区域，才可以进行全景拼接
    if (H)
    {
        //拼接图像，img1是左图，img2是右图
        CalcFourCorner(H, img2);//计算图2的四个角经变换后的坐标
        //为拼接结果图xformed分配空间,高度为图1图2高度的较小者，根据图2右上角和右下角变换后的点的位置决定拼接图的宽度
        xformed = cvCreateImage(cvSize(MIN(rightTop.x, rightBottom.x), MIN(img1->height, img2->height)), IPL_DEPTH_8U, 3);
        //用变换矩阵H对右图img2做投影变换(变换后会有坐标右移)，结果放到xformed中
        cvWarpPerspective(img2, xformed, H, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
        cvNamedWindow(IMG_MOSAIC_TEMP); //显示临时图,即只将图2变换后的图
        cvShowImage(IMG_MOSAIC_TEMP, xformed);
        cvWaitKey(0);


        //简易拼接法：直接将将左图img1叠加到xformed的左边
        IplImage* xformed_simple = cvCloneImage(xformed);//简易拼接图，可笼子xformed
        cvSetImageROI(xformed_simple, cvRect(0, 0, img1->width, img1->height));
        cvAddWeighted(img1, 1, xformed_simple, 0, 0, xformed_simple);
        cvResetImageROI(xformed_simple);
        cvNamedWindow(IMG_MOSAIC_SIMPLE);//创建窗口
        cvShowImage(IMG_MOSAIC_SIMPLE, xformed_simple);//显示简易拼接图
        cvWaitKey(0);


        //处理后的拼接图，克隆自xformed
        IplImage* xformed_proc = cvCloneImage(xformed);

        //重叠区域左边的部分完全取自图1
        cvSetImageROI(img1, cvRect(0, 0, MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        cvSetImageROI(xformed, cvRect(0, 0, MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        cvSetImageROI(xformed_proc, cvRect(0, 0, MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        cvAddWeighted(img1, 1, xformed, 0, 0, xformed_proc);
        cvResetImageROI(img1);
        cvResetImageROI(xformed);
        cvResetImageROI(xformed_proc);
        cvNamedWindow(IMG_MOSAIC_BEFORE_FUSION);
        cvShowImage(IMG_MOSAIC_BEFORE_FUSION, xformed_proc);//显示融合之前的拼接图
        cvWaitKey(0);

        //采用加权平均的方法融合重叠区域
        int start = MIN(leftTop.x, leftBottom.x);//开始位置，即重叠区域的左边界
        double processWidth = img1->width - start;//重叠区域的宽度
        double alpha = 1;//img1中像素的权重
        for (int i = 0; i<xformed_proc->height; i++)//遍历行
        {
            const uchar * pixel_img1 = ((uchar *)(img1->imageData + img1->widthStep * i));//img1中第i行数据的指针
            const uchar * pixel_xformed = ((uchar *)(xformed->imageData + xformed->widthStep * i));//xformed中第i行数据的指针
            uchar * pixel_xformed_proc = ((uchar *)(xformed_proc->imageData + xformed_proc->widthStep * i));//xformed_proc中第i行数据的指针
            for (int j = start; j<img1->width; j++)//遍历重叠区域的列
            {
                //如果遇到图像xformed中无像素的黑点，则完全拷贝图1中的数据
                if (pixel_xformed[j * 3] < 50 && pixel_xformed[j * 3 + 1] < 50 && pixel_xformed[j * 3 + 2] < 50)
                {
                    alpha = 1;
                }
                else
                {   //img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比
                    alpha = (processWidth - (j - start)) / processWidth;
                }
                pixel_xformed_proc[j * 3] = pixel_img1[j * 3] * alpha + pixel_xformed[j * 3] * (1 - alpha);//B通道
                pixel_xformed_proc[j * 3 + 1] = pixel_img1[j * 3 + 1] * alpha + pixel_xformed[j * 3 + 1] * (1 - alpha);//G通道
                pixel_xformed_proc[j * 3 + 2] = pixel_img1[j * 3 + 2] * alpha + pixel_xformed[j * 3 + 2] * (1 - alpha);//R通道
            }
        }
        cvNamedWindow(IMG_MOSAIC_PROC);//创建窗口
        cvShowImage(IMG_MOSAIC_PROC, xformed_proc);//显示处理后的拼接图
        cvWaitKey(0);

        //*重叠区域取两幅图像的平均值，效果不好
        //设置ROI，是包含重叠区域的矩形
        cvSetImageROI(xformed_proc, cvRect(MIN(leftTop.x, leftBottom.x), 0, img1->width - MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        cvSetImageROI(img1, cvRect(MIN(leftTop.x, leftBottom.x), 0, img1->width - MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        cvSetImageROI(xformed, cvRect(MIN(leftTop.x, leftBottom.x), 0, img1->width - MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        cvAddWeighted(img1, 0.5, xformed, 0.5, 0, xformed_proc);
        cvResetImageROI(xformed_proc);
        cvResetImageROI(img1);
        cvResetImageROI(xformed); //*/

        /*对拼接缝周围区域进行滤波来消除拼接缝，效果不好
        //在处理前后的图上分别设置横跨拼接缝的矩形ROI
        cvSetImageROI(xformed_proc,cvRect(img1->width-10,0,img1->width+10,xformed->height));
        cvSetImageROI(xformed,cvRect(img1->width-10,0,img1->width+10,xformed->height));
        cvSmooth(xformed,xformed_proc,CV_MEDIAN,5);//对拼接缝周围区域进行中值滤波
        cvResetImageROI(xformed);
        cvResetImageROI(xformed_proc);
        cvShowImage(IMG_MOSAIC_PROC,xformed_proc);//显示处理后的拼接图 */

        /*想通过锐化解决变换后的图像失真的问题，对于扭曲过大的图像，效果不好
        double a[]={  0, -1,  0, -1,  5, -1, 0, -1,  0  };//拉普拉斯滤波核的数据
        CvMat kernel = cvMat(3,3,CV_64FC1,a);//拉普拉斯滤波核
        cvFilter2D(xformed_proc,xformed_proc,&kernel);//滤波
        cvShowImage(IMG_MOSAIC_PROC,xformed_proc);//显示处理后的拼接图*/
        
    }



    cvReleaseImage(&stacked);
    cvReleaseImage(&img1);
    cvReleaseImage(&img2);
    kdtree_release(kd_root);
    free(feat1);
    free(feat2);
    return 0;
}
