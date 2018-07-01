#include <iostream>
#include <fstream>
#include <cmath>
// project file
#include "BundleAdjuster.h"
#include "Map.h"
#include "VisualOdometry.h"
// eigen file
#include <Eigen/Core>
#include <Eigen/Geometry>
// opencv file
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgproc/imgproc.hpp"
// Sophus
#include <sophus/se3.hpp>


class VisualSLAM{
private:
    BundleAdjuster BA;
    Map map;
    VisualOdometry VO;
    Eigen::Matrix3d K;
    std::vector<Sophus::SE3d> groundTruthData;


public:
    VisualSLAM();
    int getTestValueFromMap();
    //TO DO get camera intrinsics
    Eigen::Matrix3d getCameraIntrinsics(std::string camera_intrinsics_path);
    Sophus::SE3d getPose(int k );


    //TO DO estimate3D2DFrontEndWithOpicalFlow()
    Sophus::SE3d estimate3D2DFrontEndWithOpicalFlow(cv::Mat leftImage_, cv::Mat rightImage, std::vector<cv::Point2f>
            &previousFrame2DPoints, std::vector<cv::Point2f>&currFrame2DPoints,cv::Mat& previousImage,Sophus::SE3d prePose );

    void readGroundTruthData(std::string fileName, int numberFrames, std::vector<Sophus::SE3d>& groundTruthData);

    void plotTrajectoryNextStep(cv::Mat& window, int index, Eigen::Vector3d& translGTAccumulated, Eigen::Vector3d& translEstimAccumulated,
                                Sophus::SE3d groundTruthPose, Sophus::SE3d groundTruthPrevPose, Eigen::Matrix3d& cumR, Sophus::SE3d pose,
                                Sophus::SE3d prevPose = Sophus::SE3d(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0,0,0)));
};