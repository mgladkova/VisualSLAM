#include <Eigen/Core>
#include <Eigen/Dense>
#include <sophus/se3.hpp>     //2
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <pangolin/pangolin.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/stereo.hpp>
#include <opencv2/core/eigen.hpp>

#include <string>
#include <vector>
#include <unistd.h>

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
        void setReferenceFrame(const cv::Mat image, const cv::Mat disparity, const std::vector<cv::KeyPoint> keypoints, const cv::Mat descriptor);
        KeyFrame getReferenceFrame() const;

        void setPose(const Sophus::SE3d pose);
        Sophus::SE3d getPose() const;

        cv::Mat getDisparityMap(const cv::Mat image_left, const cv::Mat image_right);
        cv::Rect computeROIDisparityMap(cv::Size2i src_sz, cv::Ptr<cv::stereo::StereoBinarySGBM> matcher_instance);

        void extractORBFeatures(cv::Mat frame_new, std::vector<cv::KeyPoint>& keypoints_new, cv::Mat& descriptors_new);
        std::vector<cv::DMatch> findGoodORBFeatureMatches(std::vector<cv::KeyPoint> keypoints_new, cv::Mat descriptors_new);

        void get3D2DCorrespondences(std::vector<cv::KeyPoint> keypoints_new, std::vector<cv::DMatch> matches, std::vector<cv::Point3d>& p3d, std::vector<cv::Point2d>& p2d,
                                    cv::Mat disparity_map, Eigen::Matrix3d K);
        void get2D2DCorrespondences(std::vector<cv::KeyPoint> keypoints_new, std::vector<cv::DMatch> matches, std::vector<cv::Point2d>& p2d_1, std::vector<cv::Point2d>& p2d_2);

        std::vector<int> estimatePose3D2D(std::vector<cv::Point3d> p3d, std::vector<cv::Point2d> p2d, Eigen::Matrix3d K);
        void estimatePose2D2D(std::vector<cv::Point2d> p2d_1, std::vector<cv::Point2d> p2d_2, Eigen::Matrix3d K);

        void trackFeatures();
        void computeAndShowPointCloud(const cv::Mat image_left, const cv::Mat disparity, const float baseline, Eigen::Matrix3d K);

};
