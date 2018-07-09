#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/stereo.hpp>
#include <opencv2/core/eigen.hpp>

#include <string>
#include <vector>

struct KeyFrame {
	cv::Mat image;
    cv::Mat disparity_map;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;
};

class VisualOdometry {
	private:
        KeyFrame refFrame;
        Sophus::SE3d pose;

	public:
		VisualOdometry();
        void setReferenceFrame(const cv::Mat image, const cv::Mat disparity_map, const std::vector<cv::KeyPoint> keypoints, const cv::Mat descriptors);
        KeyFrame getReferenceFrame() const;

        void setPose(const Sophus::SE3d pose);
        Sophus::SE3d getPose() const;

        void setKeyFrameKeypoints(std::vector<cv::KeyPoint> updatedKeypoints);
        std::vector<cv::Point2d> get2DPointsKeyFrame();

        std::vector<cv::Point3f> get3DCoordinates(std::vector<cv::Point2f> points2D, cv::Mat disparity_map, Eigen::Matrix3d K, std::vector<uchar>& status);
        std::vector<cv::Point3f> get3DCoordinates(std::vector<cv::KeyPoint> keypoints, cv::Mat disparity_map, Eigen::Matrix3d K, std::vector<uchar>& status);

        cv::Mat getDisparityMap(const cv::Mat image_left, const cv::Mat image_right);
        cv::Rect computeROIDisparityMap(cv::Size2i src_sz, cv::Ptr<cv::stereo::StereoBinarySGBM> matcher_instance);

        void extractORBFeatures(cv::Mat frame_new, std::vector<cv::KeyPoint>& keypoints_new, cv::Mat& descriptors_new);

        std::vector<cv::DMatch> findGoodORBFeatureMatches(std::vector<cv::KeyPoint> keypointsPrevFrame, std::vector<cv::KeyPoint> keypointsCurrentFrame,
                                                          cv::Mat descriptorsPrevFrame, cv::Mat descriptorsCurrentFrame);

        void get2D2DCorrespondences(std::vector<cv::KeyPoint> keypointsPrevFrame, std::vector<cv::KeyPoint> keypointsCurrentFrame, std::vector<cv::DMatch> matches, std::vector<cv::Point2f>& p2dPrevFrame, std::vector<cv::Point2f>& p2dCurrentFrame);

        void estimatePose3D2D(std::vector<cv::Point3f>& p3d, std::vector<cv::Point2f>& p2d_PrevFrame, std::vector<cv::Point2f>& p2d_CurrentFrame, std::vector<int>& indices,  Eigen::Matrix3d K, Sophus::SE3d& pose);
        void estimatePose2D2D(std::vector<cv::Point2f> p2d_1, std::vector<cv::Point2f> p2d_2, Eigen::Matrix3d K, Sophus::SE3d& pose);

        std::vector<uchar> trackFeatures(const cv::Mat prevFrame, const cv::Mat currFrame, std::vector<cv::Point2f>& currFramePoints, std::vector<cv::Point2f>& prevFramePoints,
                           const int thresholdNumberFeatures, bool& initialize);
};
