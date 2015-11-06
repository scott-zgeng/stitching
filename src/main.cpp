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



mv_point_t leftTop, leftBottom, rightTop, rightBottom;

void CalcFourCorner(mv_mat_handle H, mv_image_t* img2);




int image_stitching()
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

    IplImage* org1 = cvLoadImage(imgfile1);
    if (!org1) {
        WRITE_ERROR_LOG("unable to load image from %s", imgfile1);
        return 1;
    }

    IplImage* org2 = cvLoadImage(imgfile2);
    if (!org2) {
        WRITE_ERROR_LOG("unable to load image from %s", imgfile2);
        return 1;
    }


    mv_image_t* img1 = mv_image_cv2mv(org1);
    if (!img1) {
        WRITE_ERROR_LOG("unable to load image from %s", imgfile1);
        return 1;
    }

    mv_image_t* img2 = mv_image_cv2mv(org2);    
    if (!img2) {
        WRITE_ERROR_LOG("unable to load image from %s", imgfile2);
        return 1;
    }


    //stacked = stack_imgs(img1, img2);

    WRITE_INFO_LOG("Finding features in %s...", imgfile1);
    sift_runtime* sift1 = new sift_runtime();
    sift_runtime::mv_features features1;
    sift1->process(img1, &features1);
    
    sift_runtime* sift2 = new sift_runtime();
    sift_runtime::mv_features features2;
    WRITE_INFO_LOG("Finding features in %s...", imgfile2);
    sift2->process(img2, &features2);


    WRITE_INFO_LOG("Building kd tree...");
    kd_root = kdtree_build(features1[0], features1.size());

    for (i = 0; i < features2.size(); i++)
    {
        feat = features2[i];
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
                features2[i]->fwd_match = nbrs[0];
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


    mv_mat_handle H;
    mv_image_t* xformed;
    struct feature **inliers;
    int n_inliers;

    H = ransac_xform(features2[0], features2.size(), FEATURE_FWD_MATCH, lsq_homog, 4, 0.01, homog_xfer_err, 3.0, &inliers, &n_inliers);
    //mv_image_t *stacked_ransac;

    //若能成功计算出变换矩阵，即两幅图中有共同区域
    if (H)
    {        
        //stacked_ransac = stack_imgs_horizontal(img1, img2);//合成图像，显示经RANSAC算法筛选后的匹配结果

        //img1LeftBound = inliers[0]->fwd_match->x;//图1中匹配点外接矩形的左边界
        //img1RightBound = img1LeftBound;//图1中匹配点外接矩形的右边界
        //img2LeftBound = inliers[0]->x;//图2中匹配点外接矩形的左边界
        //img2RightBound = img2LeftBound;//图2中匹配点外接矩形的右边界

        int invertNum = 0;//统计pt2.x > pt1.x的匹配点对的个数，来判断img1中是否右图

        //遍历经RANSAC算法筛选后的特征点集合inliers，找到每个特征点的匹配点，画出连线
        for (int i = 0; i<n_inliers; i++)
        {
            feat = inliers[i];//第i个特征点
            pt2 = mv_point_t(mv_round(feat->x), mv_round(feat->y));//图2中点的坐标
            pt1 = mv_point_t(mv_round(feat->fwd_match->x), mv_round(feat->fwd_match->y));//图1中点的坐标(feat的匹配点)
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
            //mv_line(stacked_ransac, pt1, pt2, MV_RGB(255, 0, 255), 1, 8, 0);//在匹配图上画出连线
        }

        //绘制图1中包围匹配点的矩形
        //cvRectangle(stacked_ransac,cvPoint(img1LeftBound,0),cvPoint(img1RightBound,img1->height),MV_RGB(0,255,0),2);
        //绘制图2中包围匹配点的矩形
        //cvRectangle(stacked_ransac,cvPoint(img1->width+img2LeftBound,0),cvPoint(img1->width+img2RightBound,img2->height),MV_RGB(0,0,255),2);

        //mv_named_window(IMG_MATCH2);//创建窗口
        //mv_show_image(IMG_MATCH2, stacked_ransac);//显示经RANSAC算法筛选后的匹配图
        //mv_wait_key(0);


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
            mv_mat_handle H_IVT = mv_create_matrix(3, 3);//变换矩阵的逆矩阵
            //求H的逆阵H_IVT时，若成功求出，返回非零值
            if (mv_invert(H, H_IVT))
            {
                
                mv_release_matrix(H);//释放变换矩阵H，因为用不到了
                H = mv_clone_matrix(H_IVT);//将H的逆阵H_IVT中的数据拷贝到H中
                mv_release_matrix(H_IVT);//释放逆阵H_IVT
                //将img1和img2对调
                mv_image_t * temp = img2;
                img2 = img1;
                img1 = temp;
                //cvShowImage(IMG1,img1);
                //cvShowImage(IMG2,img2);
                //ui->mosaicButton->setEnabled(true);//激活全景拼接按钮
            }
            else//H不可逆时，返回0
            {
                mv_release_matrix(H_IVT);//释放逆阵H_IVT
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
        xformed = mv_create_image(mv_size_t(MIN(rightTop.x, rightBottom.x), MIN(img1->height, img2->height)), IPL_DEPTH_8U, 3);
        //用变换矩阵H对右图img2做投影变换(变换后会有坐标右移)，结果放到xformed中
        mv_warp_perspective(img2, xformed, H, MV_INTER_LINEAR + MV_WARP_FILL_OUTLIERS, mv_scalar_t(0));
        cvNamedWindow(IMG_MOSAIC_TEMP); //显示临时图,即只将图2变换后的图
        //cvShowImage(IMG_MOSAIC_TEMP, xformed);
        cvWaitKey(0);


        //简易拼接法：直接将将左图img1叠加到xformed的左边
        mv_image_t* xformed_simple = mv_clone_image(xformed);//简易拼接图，可笼子xformed
        mv_set_image_roi(xformed_simple, mv_rect_t(0, 0, img1->width, img1->height));
        mv_add_weighted(img1, 1, xformed_simple, 0, 0, xformed_simple);
        mv_reset_image_roi(xformed_simple);
        cvNamedWindow(IMG_MOSAIC_SIMPLE);//创建窗口
        //cvShowImage(IMG_MOSAIC_SIMPLE, xformed_simple);//显示简易拼接图
        cvWaitKey(0);


        //处理后的拼接图，克隆自xformed
        mv_image_t* xformed_proc = mv_clone_image(xformed);

        //重叠区域左边的部分完全取自图1
        mv_set_image_roi(img1, mv_rect_t(0, 0, MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        mv_set_image_roi(xformed, mv_rect_t(0, 0, MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        mv_set_image_roi(xformed_proc, mv_rect_t(0, 0, MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        mv_add_weighted(img1, 1, xformed, 0, 0, xformed_proc);
        mv_reset_image_roi(img1);
        mv_reset_image_roi(xformed);
        mv_reset_image_roi(xformed_proc);
        cvNamedWindow(IMG_MOSAIC_BEFORE_FUSION);
        //cvShowImage(IMG_MOSAIC_BEFORE_FUSION, xformed_proc);//显示融合之前的拼接图
        cvWaitKey(0);

        //采用加权平均的方法融合重叠区域
        int start = MIN(leftTop.x, leftBottom.x);//开始位置，即重叠区域的左边界
        double processWidth = img1->width - start;//重叠区域的宽度
        double alpha = 1;//img1中像素的权重
        for (int i = 0; i<xformed_proc->height; i++)//遍历行
        {
            const unsigned char * pixel_img1 = ((unsigned char *)(img1->imageData + img1->widthStep * i));//img1中第i行数据的指针
            const unsigned char * pixel_xformed = ((unsigned char *)(xformed->imageData + xformed->widthStep * i));//xformed中第i行数据的指针
            unsigned char * pixel_xformed_proc = ((unsigned char *)(xformed_proc->imageData + xformed_proc->widthStep * i));//xformed_proc中第i行数据的指针
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
        //cvShowImage(IMG_MOSAIC_PROC, xformed_proc);//显示处理后的拼接图
        cvWaitKey(0);

        //*重叠区域取两幅图像的平均值，效果不好
        //设置ROI，是包含重叠区域的矩形
        mv_set_image_roi(xformed_proc, mv_rect_t(MIN(leftTop.x, leftBottom.x), 0, img1->width - MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        mv_set_image_roi(img1, mv_rect_t(MIN(leftTop.x, leftBottom.x), 0, img1->width - MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        mv_set_image_roi(xformed, mv_rect_t(MIN(leftTop.x, leftBottom.x), 0, img1->width - MIN(leftTop.x, leftBottom.x), xformed_proc->height));
        mv_add_weighted(img1, 0.5, xformed, 0.5, 0, xformed_proc);
        mv_reset_image_roi(xformed_proc);
        mv_reset_image_roi(img1);
        mv_reset_image_roi(xformed); //*/

        /*对拼接缝周围区域进行滤波来消除拼接缝，效果不好
        //在处理前后的图上分别设置横跨拼接缝的矩形ROI
        cvSetImageROI(xformed_proc,cvRect(img1->width-10,0,img1->width+10,xformed->height));
        cvSetImageROI(xformed,cvRect(img1->width-10,0,img1->width+10,xformed->height));
        cvSmooth(xformed,xformed_proc,MV_MEDIAN,5);//对拼接缝周围区域进行中值滤波
        cvResetImageROI(xformed);
        cvResetImageROI(xformed_proc);
        cvShowImage(IMG_MOSAIC_PROC,xformed_proc);//显示处理后的拼接图 */

        /*想通过锐化解决变换后的图像失真的问题，对于扭曲过大的图像，效果不好
        double a[]={  0, -1,  0, -1,  5, -1, 0, -1,  0  };//拉普拉斯滤波核的数据
        CvMat kernel = cvMat(3,3,MV_64FC1,a);//拉普拉斯滤波核
        cvFilter2D(xformed_proc,xformed_proc,&kernel);//滤波
        cvShowImage(IMG_MOSAIC_PROC,xformed_proc);//显示处理后的拼接图*/
        
    }


    //mv_release_image(&stacked);
    mv_release_image(&img1);
    mv_release_image(&img2);
    kdtree_release(kd_root);
    //free(feat1);
    //free(feat2);
    return 0;
}



//计算图2的四个角经矩阵H变换后的坐标
void CalcFourCorner(mv_mat_handle H, mv_image_t* img2)
{
    //计算图2的四个角经矩阵H变换后的坐标
    double v2[] = { 0, 0, 1 };//左上角
    double v1[3];//变换后的坐标值
    mv_mat_handle V2 = mv_create_matrix(3, 1);// , v2);
    mv_mat_handle V1 = mv_create_matrix(3, 1);//, v1);
    mv_matrix_mul(H, &V2, &V1);//矩阵乘法
    leftTop.x = mv_round(v1[0] / v1[2]);
    leftTop.y = mv_round(v1[1] / v1[2]);
    //cvCircle(xformed,leftTop,7,MV_RGB(255,0,0),2);

    //将v2中数据设为左下角坐标
    v2[0] = 0;
    v2[1] = img2->height;
    //V2 = mv_mat_handle(3, 1, MV_64FC1, v2);
    //V1 = mv_mat_handle(3, 1, MV_64FC1, v1);
    mv_matrix_mul(H, &V2, &V1);
    leftBottom.x = mv_round(v1[0] / v1[2]);
    leftBottom.y = mv_round(v1[1] / v1[2]);
    //cvCircle(xformed,leftBottom,7,MV_RGB(255,0,0),2);

    //将v2中数据设为右上角坐标
    v2[0] = img2->width;
    v2[1] = 0;
    //V2 = mv_mat_handle(3, 1, MV_64FC1, v2);
    //V1 = mv_mat_handle(3, 1, MV_64FC1, v1);
    mv_matrix_mul(H, &V2, &V1);
    rightTop.x = mv_round(v1[0] / v1[2]);
    rightTop.y = mv_round(v1[1] / v1[2]);
    //cvCircle(xformed,rightTop,7,MV_RGB(255,0,0),2);

    //将v2中数据设为右下角坐标
    v2[0] = img2->width;
    v2[1] = img2->height;
    //V2 = mv_mat_handle(3, 1, MV_64FC1, v2);
    //V1 = mv_mat_handle(3, 1, MV_64FC1, v1);
    mv_matrix_mul(H, &V2, &V1);
    rightBottom.x = mv_round(v1[0] / v1[2]);
    rightBottom.y = mv_round(v1[1] / v1[2]);
    //cvCircle(xformed,rightBottom,7,MV_RGB(255,0,0),2);

}


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



extern "C" {
#include "meschach/matrix.h"
#include "meschach/matrix2.h"
}


void test_matrix()
{
    MAT* new_m = m_get(2, 2);
    
    m_set_val(new_m, 0, 0, 1);
    m_set_val(new_m, 0, 1, 2);
    m_set_val(new_m, 1, 0, 3);
    m_set_val(new_m, 1, 1, 4);

    MAT* result = m_get(2, 2);
    m_inverse(new_m, result);

    VEC* out_v = v_get(2);
    
    MAT* new_n = m_get(2, 2);
    m_set_val(new_n, 0, 0, 4);
    m_set_val(new_n, 0, 1, 3);
    m_set_val(new_n, 1, 0, 2);
    m_set_val(new_n, 1, 1, 1);
    svd(new_m, new_n, NULL, out_v);
    WRITE_INFO_LOG("%f, %f", out_v->ve[0], out_v->ve[1]);
    
    CvMat* H = cvCreateMat(2, 2, CV_64FC1);
    cvmSet(H, 0, 0, 1);
    cvmSet(H, 0, 1, 2);
    cvmSet(H, 1, 0, 3);
    cvmSet(H, 1, 1, 4);
    

    CvMat* V = cvCreateMat(2, 2, CV_64FC1);
    cvmSet(V, 0, 0, 4);
    cvmSet(V, 0, 1, 3);
    cvmSet(V, 1, 0, 2);
    cvmSet(V, 1, 1, 1);

    CvMat* R = cvCreateMat(2, 2, CV_64FC1);
    cvSolve(H, V, R, CV_SVD);
    WRITE_INFO_LOG("%f, %f", cvmGet(R, 0, 0), cvmGet(R, 0, 1));
    WRITE_INFO_LOG("%f, %f", cvmGet(R, 1, 0), cvmGet(R, 1, 1));
    
    WRITE_INFO_LOG("s");

    //m_add()	        Add matrices
    //m_mlt()	        Multiplies matrices
    //m_sub()		    Subtract matrices
    //mv_mlt()	        Computes  Ax
    //mv_mltadd()	    Computes  y <-Ax + y
    //m_zero()	        Zero a matrix

}


int main(int argc, char** argv)
{
    //image_stitching();
    //test_image_proc();
    test_matrix();

    return 0;
}




