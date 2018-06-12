// -------------- test the visual odometry -------------
#include <fstream>
#include <boost/timer.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp> 

// additional include files
#include <Eigen/Core>
#include <Eigen/Dense>
#include <sophus/se3.h>
#include <opencv2/ximgproc/disparity_filter.hpp>
//#include <pangolin/pangolin.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/stereo.hpp>
#include <opencv2/core/eigen.hpp>

#include <string>
#include <vector>
#include <unistd.h>
using namespace cv;
using namespace cv::ximgproc;
using namespace std;

//

#include "myslam/config.h"
#include "myslam/visual_odometry.h"



//cv::Rect computeROIDisparityMap(cv::Size2i src_sz, cv::Ptr<cv::stereo::StereoBinarySGBM> matcher_instance)
//{
//    int min_disparity = matcher_instance->getMinDisparity();
//    int num_disparities = matcher_instance->getNumDisparities();
//    int block_size = matcher_instance->getBlockSize();

//    int bs2 = block_size/2;
//    int minD = min_disparity, maxD = min_disparity + num_disparities - 1;

//    int xmin = maxD + bs2;
//    int xmax = src_sz.width + minD - bs2;
//    int ymin = bs2;
//    int ymax = src_sz.height - bs2;

//    cv::Rect r(xmin, ymin, xmax - xmin, ymax - ymin);
//    return r;
//}

//cv::Mat getDisparityMap(const cv::Mat image_left, const cv::Mat image_right){
//    cv::Mat disparity, true_dmap, disparity_norm;
//    cv::Rect ROI;
//    int min_disparity = 0;
//    int number_of_disparities = 16*6 - min_disparity;
//    int kernel_size = 7;

//     std::cout<<"error"<<std::endl;
//    cv::Ptr<cv::stereo::StereoBinarySGBM> sgbm = cv::stereo::StereoBinarySGBM::create(min_disparity, number_of_disparities, kernel_size);
//    // setting the penalties for sgbm
//    sgbm->setP1(8*std::pow(kernel_size, 2));
//    sgbm->setP2(32*std::pow(kernel_size, 2));
//    sgbm->setMinDisparity(min_disparity);
//    sgbm->setUniquenessRatio(3);
//    sgbm->setSpeckleWindowSize(200);
//    sgbm->setSpeckleRange(32);
//    sgbm->setDisp12MaxDiff(1);
//    sgbm->setSpekleRemovalTechnique(cv::stereo::CV_SPECKLE_REMOVAL_AVG_ALGORITHM);
//    sgbm->setSubPixelInterpolationMethod(cv::stereo::CV_SIMETRICV_INTERPOLATION);

//    // setting the penalties for sgbm
//    ROI = computeROIDisparityMap(image_left.size(),sgbm);
//    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
//    wls_filter = cv::ximgproc::createDisparityWLSFilterGeneric(false);
//    wls_filter->setDepthDiscontinuityRadius(1);

//    sgbm->compute(image_left, image_right, disparity);
//    wls_filter->setLambda(8000.0);
//    wls_filter->setSigmaColor(1.5);
//    wls_filter->filter(disparity,image_left,disparity_norm,cv::Mat(), ROI);

//    cv::Mat filtered_disp_vis;
//    cv::ximgproc::getDisparityVis(disparity_norm,filtered_disp_vis,1);
//    /*
//    cv::namedWindow("filtered disparity", cv::WINDOW_AUTOSIZE);
//    cv::imshow("filtered disparity", filtered_disp_vis);
//    cv::waitKey();
//    */
//    filtered_disp_vis.convertTo(true_dmap, CV_32F, 1.0/16.0, 0.0);
//    return true_dmap;
//}

cv::Rect computeROI(cv::Size2i src_sz, cv::Ptr<cv::StereoMatcher> matcher_instance)
{
    int min_disparity = matcher_instance->getMinDisparity();
    int num_disparities = matcher_instance->getNumDisparities();
    int block_size = matcher_instance->getBlockSize();

    int bs2 = block_size/2;
    int minD = min_disparity, maxD = min_disparity + num_disparities - 1;

    int xmin = maxD + bs2;
    int xmax = src_sz.width + minD - bs2;
    int ymin = bs2;
    int ymax = src_sz.height - bs2;

    cv::Rect r(xmin, ymin, xmax - xmin, ymax - ymin);
    return r;
}

cv::Mat getDisparityMap(const cv::Mat image_left, const cv::Mat image_right){
    cv::Mat disparity, disparity_norm;
    cv::Rect ROI;
    int number_of_disparities = 16*8;
    int kernel_size = 3;

//    cv::Ptr<cv::stereo::StereoBinarySGBM> sgbm = cv::stereo::StereoBinarySGBM::create(0, number_of_disparities, kernel_size);
    Ptr<StereoSGBM> sgbm  = StereoSGBM::create(0, number_of_disparities, kernel_size);
    // setting the penalties for sgbm
    sgbm->setP1(10);
    sgbm->setP2(240);
    sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
    ROI = computeROI(image_left.size(),sgbm);
    cv::Ptr<DisparityWLSFilter> wls_filter;
    wls_filter = createDisparityWLSFilterGeneric(false);
    wls_filter->setDepthDiscontinuityRadius((int)ceil(0.5*3));
//    sgbm->setPreFilterCap(63); //
//    sgbm->setMinDisparity(4);
//    sgbm->setUniquenessRatio(4);
//    sgbm->setSpeckleWindowSize(45);
//    sgbm->setSpeckleRange(2);
//    sgbm->setDisp12MaxDiff(1);
//    sgbm->setBinaryKernelType(CV_8UC1);
//    sgbm->setSpekleRemovalTechnique(cv::stereo::CV_SPECKLE_REMOVAL_AVG_ALGORITHM);
//    sgbm->setSubPixelInterpolationMethod(cv::stereo::CV_SIMETRICV_INTERPOLATION);
    sgbm->compute(image_left, image_right, disparity);
    wls_filter->setLambda(8000.0);
    wls_filter->setSigmaColor(1.5);
//    filtering_time = (double)getTickCount();
    wls_filter->filter(disparity,image_left,disparity_norm,Mat(),ROI);
    Mat filtered_disp_vis;
    getDisparityVis(disparity_norm,filtered_disp_vis,1);
//    namedWindow("filtered disparity", WINDOW_AUTOSIZE);
//    imshow("filtered disparity", filtered_disp_vis);
//    waitKey();
    return filtered_disp_vis;
}


//int add()
//{
//    return 5;
//}

int main ( int argc, char** argv )
{
//    cv::Mat image_left=cv::imread("/home/fyl/Desktop/vision_based_navigation/final_project/about_git/VisualSLAM-feature_second_draft/data/left/000000.png");
//    cv::Mat image_right=cv::imread("/home/fyl/Desktop/vision_based_navigation/final_project/about_git/VisualSLAM-feature_second_draft/data/right/000000.png");
//    cv::Mat disparity_map = getDisparityMap(image_left, image_right);

////        cv::namedWindow("depth", cv::WINDOW_AUTOSIZE);
////        cv::imshow("depth_map", depth);
//    cv::imshow("color", image_left);

//    cv::imshow("color_right", image_right);
//    cv::imshow("depth", disparity_map);
//    std::cout<<add()<<std::endl;
//    cv::waitKey();

//    while(1){}


    if ( argc != 2 )
    {
        cout<<"usage: run_vo parameter_file"<<endl;
        return 1;
    }

    myslam::Config::setParameterFile ( argv[1] );
    myslam::VisualOdometry::Ptr vo ( new myslam::VisualOdometry );

    string dataset_dir = myslam::Config::get<string> ( "dataset_dir" );
    cout<<"dataset: "<<dataset_dir<<endl;
    ifstream fin ( dataset_dir+"/associate.txt" );
    if ( !fin )
    {
        cout<<"please generate the associate file called associate.txt!"<<endl;
        return 1;
    }

    vector<string> rgb_files_left,rgb_files_right, depth_files;
    vector<double> rgb_times, depth_times;
    while ( !fin.eof() )
    {
        string rgb_time, rgb_file_left,rgb_file_right, depth_time, depth_file;
        fin>>rgb_file_left>>rgb_file_right;//>>depth_time>>depth_file;
//        rgb_times.push_back ( atof ( rgb_time.c_str() ) );
        depth_times.push_back ( atof ( depth_time.c_str() ) );
        rgb_files_left.push_back ( dataset_dir+"/"+rgb_file_left );
        rgb_files_right.push_back ( dataset_dir+"/"+rgb_file_right );
//        depth_files.push_back ( dataset_dir+"/"+depth_file );

        if ( fin.good() == false )
            break;
    }

    myslam::Camera::Ptr camera ( new myslam::Camera );
    
    // visualization
    cv::viz::Viz3d vis("Visual Odometry");
    cv::viz::WCoordinateSystem world_coor(1.0), camera_coor(0.5);
    cv::Point3d cam_pos( 0, -1.0, -1.0 ), cam_focal_point(0,0,0), cam_y_dir(0,1,0);
    cv::Affine3d cam_pose = cv::viz::makeCameraPose( cam_pos, cam_focal_point, cam_y_dir );
    vis.setViewerPose( cam_pose );
    
    world_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
    camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0);
    vis.showWidget( "World", world_coor );
    vis.showWidget( "Camera", camera_coor );

    cout<<"read total "<<rgb_files_left.size() <<" entries"<<endl;
    for ( int i=0; i<rgb_files_left.size(); i++ )
    {
//    int i=0;
        Mat color = cv::imread ( rgb_files_left[i] );
        Mat color_right = cv::imread ( rgb_files_right[i] );
        std::cout<<rgb_files_right[i]<<std::endl;
//        Mat depth = cv::imread ( depth_files[i], -1 );
        Mat depth = getDisparityMap(color,color_right);

        /*  test depth image*/
        /*cv::imshow("color", color);

        cv::imshow("color_right", color_right);
        cv::imshow("depth", disparity_map);
        cv::waitKey();
        */


        if ( color.data==nullptr || depth.data==nullptr )
            break;
        myslam::Frame::Ptr pFrame = myslam::Frame::createFrame();
        pFrame->camera_ = camera;
        pFrame->color_ = color;
        pFrame->depth_ = depth;
//        pFrame->time_stamp_ = rgb_times[i];

        boost::timer timer;
        vo->addFrame ( pFrame );
        cout<<"VO costs time: "<<timer.elapsed()<<endl;
        
        if ( vo->state_ == myslam::VisualOdometry::LOST )
            break;
        SE3 Tcw = pFrame->T_c_w_.inverse();
        // cout<<Tcw.matrix()<<endl;
        cout<< pFrame->T_c_w_.matrix()<<endl;
        
        // show the map and the camera pose
        cv::Affine3d M(
            cv::Affine3d::Mat3(
                Tcw.rotation_matrix()(0,0), Tcw.rotation_matrix()(0,1), Tcw.rotation_matrix()(0,2),
                Tcw.rotation_matrix()(1,0), Tcw.rotation_matrix()(1,1), Tcw.rotation_matrix()(1,2),
                Tcw.rotation_matrix()(2,0), Tcw.rotation_matrix()(2,1), Tcw.rotation_matrix()(2,2)
            ),
            cv::Affine3d::Vec3(
                Tcw.translation()(0,0), Tcw.translation()(1,0), Tcw.translation()(2,0)
            )
        );
        
        cv::imshow("image", color );
        cv::waitKey(1);
        vis.setWidgetPose( "Camera", M);
        vis.spinOnce(1, false);
    }

    return 0;
}
