#include "Map.h"
#include "BundleAdjuster.h"
#include "VisualOdometry.h"

#include <iostream>     // std::cout
#include <fstream>      // std::ifstream

class VisualSLAM {
private:
	Map map;
        BundleAdjuster BA;
//	VisualOdometry VO;

public:
	VisualSLAM();
	void readCameraIntrisics(std::string camera_intrinsics_file);
	void performFrontEndStep(cv::Mat left_image, cv::Mat right_image); // feature detection / tracking and matching 
	void runBackEndRoutine(); // optimization over parameters.numImagesBundleAdjustment images
	void update(); // update map and poses
        VisualOdometry VO;

};
