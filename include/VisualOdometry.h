#include <iostream>
#include <vector>
// opencv
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>
#include <opencv2/stereo.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
//#include <opencv/cxeigen.hpp>
#include <sophus/so3.h>
#include <opencv2/core/eigen.hpp>
// eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
//sophus
#include <sophus/se3.h>

class VisualOdometry{
private:
//    std::vector<unsigned char> status;
    float baseline = 0.53716;
    cv::Mat disparityMap;
//    std::vector<Sophus::SE3> historyPose;
public:
    VisualOdometry() {};

    //TO DO harrisDection
    //TO DO featureTracking
    std::vector<uchar> corr2DPointsFromPreFrame2DPoints(cv::Mat previousImage, cv::Mat currImage,
                                                                std::vector<cv::Point2f>& previousFrame2DPoints,
                                                                std::vector<cv::Point2f>& currFrame2DPoints);
    //TO DO getDisparityMap
    cv::Rect computeROIDisparityMap(cv::Size2i src_sz, cv::Ptr<cv::stereo::StereoBinarySGBM> matcher_instance);
    void generateDisparityMap(const cv::Mat image_left, const cv::Mat image_right);
    //TO DO getDepth  currFrame2DPoints
    std::vector<cv::Point3f> getDepth3DPointsFromCurrImage(std::vector<cv::Point2f>& currFrame2DPoints,Eigen::Matrix3d K);

    //TO DO poseEstimate2D3DPnp
    Sophus::SE3 poseEstimate2D3DPNP(std::vector<cv::Point3f>& p3d, std::vector<cv::Point2f>& p2d,Eigen::Matrix3d K);

};