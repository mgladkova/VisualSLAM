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
#include <sophus/se3.h>


class VisualSLAM{
private:
    BundleAdjuster BA;
    Map map;
    VisualOdometry VO;
    Eigen::Matrix3d K;
    std::vector<Sophus::SE3> groundTruthData;


public:
    VisualSLAM();
    int getTestValueFromMap();
    //TO DO get camera intrinsics
    Eigen::Matrix3d getCameraIntrinsics(std::string camera_intrinsics_path);
    Sophus::SE3 getPose(int k );


    //TO DO estimate3D2DFrontEndWithOpicalFlow()
    Sophus::SE3 estimate3D2DFrontEndWithOpicalFlow(cv::Mat leftImage_, cv::Mat rightImage, std::vector<cv::Point2f>
            &previousFrame2DPoints, std::vector<cv::Point2f>&currFrame2DPoints,cv::Mat& previousImage);

    void readGroundTruthData(std::string fileName, int numberFrames, std::vector<Sophus::SE3>& groundTruthData);

    void plotTrajectoryNextStep(cv::Mat& window, int index, Eigen::Vector3d& translGTAccumulated, Eigen::Vector3d& translEstimAccumulated,
                                Sophus::SE3 groundTruthPose, Sophus::SE3 groundTruthPrevPose, Eigen::Matrix3d& cumR, Sophus::SE3 pose,
                                Sophus::SE3 prevPose = Sophus::SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0,0,0)));
};