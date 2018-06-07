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

//#include "BundleAdjuster.h"

struct Camera {
	double fx;
	double fy;
	double cx;
	double cy;
};

struct KeyFrame {
	cv::Mat image;
        cv::Mat disparity_map;
        std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptor;
};

class VisualOdometry {
	private:
                KeyFrame refFrame;
		Camera cam;
		Sophus::SE3 pose;

	public:
		VisualOdometry();
                void setReferenceFrame(const cv::Mat image, const cv::Mat disparity, const std::vector<cv::KeyPoint> keypoints, const cv::Mat descriptor);
                void setCamera(double fx, double fy, double cx, double cy);
		
                KeyFrame getReferenceFrame() const;
		Camera getCamera(void) const;
		cv::Mat getDisparityMap(const cv::Mat image_left, const cv::Mat image_right);

                void extractORBFeatures(cv::Mat frame_new, std::vector<cv::KeyPoint>& keypoints_new, cv::Mat& descriptors_new);
                std::vector<cv::Point3f> estimate3DPoints(std::vector<cv::KeyPoint> keypoints, cv::Mat disparity_map);
                void Motion_BA(std::vector<cv::Point3d> p3d,std::vector<cv::Point2d> p2d,Eigen::Matrix3d K,Sophus::SE3 pose,int iteration_times);
                std::vector<cv::DMatch> findGoodORBFeatureMatches(std::vector<cv::KeyPoint> keypoints_new, cv::Mat descriptors_new);
                void estimatePose3D2D(std::vector<cv::KeyPoint> keypoints, std::vector<cv::DMatch> matches);
                void estimatePose2D2D(std::vector<cv::KeyPoint> keypoints, std::vector<cv::DMatch> matches);
                void trackFeatures();
};
