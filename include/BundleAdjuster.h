// * Class BundleAdjuster is responsible for 3D structure and camera pose optimizations


#include <Eigen/Core>
#include <Eigen/Dense>
#include <sophus/se3.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/stereo.hpp>
#include <opencv2/core/eigen.hpp>

#include <string>
#include <vector>

class BundleAdjuster {

public:
        BundleAdjuster();
        void Motion_BA(std::vector<cv::Point3d> p3d,std::vector<cv::Point2d> p2d,Eigen::Matrix3d K,Sophus::SE3 pose,int iteration_times);

};
