#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <sophus/se3.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <pangolin/pangolin.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/stereo.hpp>
#include <opencv2/core/eigen.hpp>

#include <unistd.h>
#include <mutex>
#include <condition_variable>

class VisualizationToolkit{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VisualizationToolkit(Eigen::Matrix3d calibMat, float baseline);
    void computeAndShowPointCloud();
    void getDataForPointCloudVisualization(cv::Mat& image_left, cv::Mat& disparity);
    void setDataForPointCloudVisualization(cv::Mat image_left, cv::Mat disparity);
    void visualizeAllPoses(std::vector<Sophus::SE3d> historyPoses, Eigen::Matrix3d K);
    void plot2DPoints(cv::Mat image, std::vector<cv::Point2f> points2d);
    void plot2DPoints(cv::Mat image, std::vector<cv::KeyPoint> keypoints);
    void plotTrajectoryNextStep(cv::Mat& window, int index,
                                Eigen::Vector3d& translGTAccumulated,
                                Eigen::Vector3d& translEstimAccumulatedLeft,
                                Eigen::Vector3d& translEstimAccumulatedRight,
                                Sophus::SE3d groundTruthPose, Sophus::SE3d groundTruthPrevPose,
                                Sophus::SE3d estimPoseLeft, Sophus::SE3d estimPoseRight,
                                Sophus::SE3d estimPrevPoseLeft, Sophus::SE3d estimPrevPoseRight);

    void replotTrajectory(cv::Mat& window, int index, Eigen::Vector3d& translGTAccumulated,
                          Eigen::Vector3d& translEstimAccumulatedLeft, Eigen::Vector3d& translEstimAccumulatedRight,
                          std::vector<Sophus::SE3d> cumPosesLeft, std::vector<Sophus::SE3d> cumPosesRight,
                          std::vector<Sophus::SE3d> groundTruthPoses);
    void showPointCloud(const std::vector<cv::Point3f> points3D);
private:
    Eigen::Matrix3d K;
    float baseline;

    cv::Mat image;
    cv::Mat disparity;

    std::mutex mReadWriteMutex;
    std::condition_variable mCondVar;

    bool mReadyToProcess;
    bool mProcessed;
};


