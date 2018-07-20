#include <mutex>
#include <thread>
#include <chrono>

#pragma once

#include "BundleAdjuster.h"
#include "VisualOdometry.h"

#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <algorithm>    // std::iota

class VisualSLAM {
private:
    Map map_left;
    Map map_right;
    BundleAdjuster BA;
    VisualOdometry VO;
    Eigen::Matrix3d K;

    std::vector<Sophus::SE3d> groundTruthData;

public:
	VisualSLAM();

    Sophus::SE3d getPose_left(int index);
    Sophus::SE3d getPose_right(int index);
    Sophus::SE3d getGTPose(int index) const;

    std::vector<Sophus::SE3d> getPoses_left() const;
    std::vector<Sophus::SE3d> getPoses_right() const;

    int getNumberPoses_left() const;
    int getNumberPoses_right() const;

    std::vector<cv::Point3f> getStructure3D_left() const;
    std::vector<cv::Point3f> getStructure3D_right() const;

    Eigen::Matrix3d getCameraMatrix() const;
    double getFocalLength() const;

    std::vector<std::pair<int, cv::Point2f>> getObservationsForLeftCamera(int cameraIndex);
    std::vector<std::pair<int, cv::Point2f>> getObservationsForRightCamera(int cameraIndex);

    cv::Mat getDisparityMap(cv::Mat image_left, cv::Mat image_right);
    cv::Rect computeROIDisparityMap(cv::Size2i src_sz, cv::Ptr<cv::stereo::StereoBinarySGBM> matcher_instance);

    void getDataFromImageLeftForDrawing(int& cameraIndex, Sophus::SE3d& camera, std::vector<cv::Point3f>& structure3d, std::vector<int>& obsIndices, Sophus::SE3d& gtCamera);

    void readCameraIntrisics(std::string camera_intrinsics_file);
    void readGroundTruthData(std::string fileName, int numberFrames, std::vector<Sophus::SE3d>& groundTruthData);

    Sophus::SE3d performFrontEndStep(cv::Mat image_left, cv::Mat disparity_map, std::vector<cv::KeyPoint>& keyPointsPrevFrame, cv::Mat& descriptorsPrevFrame); // feature detection / tracking and matching
    Sophus::SE3d performFrontEndStepWithTracking(cv::Mat image_left, cv::Mat disparity_map, std::vector<cv::Point2f>& pointsCurrentFrame, std::vector<cv::Point2f>& pointsPrevFrame, cv::Mat& prevImageLeft, bool isLeftImage);

    //bool performPoseGraphOptimization(int keyFrameStep, int numKeyFrames);

    bool checkPoint2DCoordinates(cv::Point2f point, cv::Mat image);
};
