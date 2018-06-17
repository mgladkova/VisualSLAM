#include "Map.h"
#include "BundleAdjuster.h"
#include "VisualOdometry.h"

#include <iostream>     // std::cout
#include <fstream>      // std::ifstream

class VisualSLAM {
private:
	Map map;
//	Map map;
    BundleAdjuster BA;
    VisualOdometry VO;
    Eigen::Matrix3d K;

    std::vector<Sophus::SE3d> historyPoses;
    std::vector<Sophus::SE3d> groundTruthData;
public:
	VisualSLAM();
    Sophus::SE3d getPose(int index);
    int getNumberPoses() const;
    Eigen::Matrix3d getCameraMatrix() const;
    double getFocalLength() const;

	void readCameraIntrisics(std::string camera_intrinsics_file);
    void readGroundTruthData(std::string fileName, int numberFrames);
	void performFrontEndStep(cv::Mat left_image, cv::Mat right_image); // feature detection / tracking and matching 
	void runBackEndRoutine(); // optimization over parameters.numImagesBundleAdjustment images
	void update(); // update map and poses
    void visualizeAllPoses();
    void visualizeTrajectoryVSGroundTruthTransformation();
    void plotTrajectoryNextStep(cv::Mat& window, Eigen::Vector3d& translGTAccumulated, Eigen::Vector3d& translEstimAccumulated);
};
