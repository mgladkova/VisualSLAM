#include "BundleAdjuster.h"
#include "VisualOdometry.h"

#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <algorithm>    // std::iota

class VisualSLAM {
private:
	Map map;
    BundleAdjuster BA;
    VisualOdometry VO;
    Eigen::Matrix3d K;

    std::vector<Sophus::SE3d> groundTruthData;
public:
	VisualSLAM();
    Sophus::SE3d getPose(int index);
    std::vector<Sophus::SE3d> getPoses() const;
    int getNumberPoses() const;
    Eigen::Matrix3d getCameraMatrix() const;
    double getFocalLength() const;

    std::vector<cv::Point3f> getStructure3D() const;

	void readCameraIntrisics(std::string camera_intrinsics_file);
    void readGroundTruthData(std::string fileName, int numberFrames, std::vector<Sophus::SE3d>& groundTruthData);
    Sophus::SE3d performFrontEndStep(cv::Mat image_left, cv::Mat image_right, std::vector<cv::KeyPoint>& keyPointsPrevFrame, cv::Mat& descriptorsPrevFrame); // feature detection / tracking and matching
    Sophus::SE3d performFrontEndStepWithTracking(cv::Mat image_left, cv::Mat image_right, std::vector<cv::Point2f>& pointsCurrentFrame, std::vector<cv::Point2f>& pointsPrevFrame, cv::Mat& prevImageLeft);
};
