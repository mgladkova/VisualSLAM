#include "VisualSLAM.h"

VisualSLAM::VisualSLAM() {}

void VisualSLAM::readCameraIntrisics(std::string camera_file_path){
	std::ifstream file;
	file.open(camera_file_path, std::ifstream::in);

	if (!file){
		throw std::runtime_error("Cannot read the file with camera intrinsics");
	}

	std::string prefix;
	double data[12];

	file >> prefix;
	for (int i = 0; i < 12; i++){
		file >> data[i];
	}

	// fx, fy, cx, cy
	VO.setCamera(data[0], data[5], data[2], data[6]);

	Camera cam = VO.getCamera();

	std::cout << cam.fx << " " << cam.fy << " " << cam.cx << " " << cam.cy << std::endl; 
}

void VisualSLAM::performFrontEndStep(cv::Mat image_left, cv::Mat image_right){
    std::vector<cv::KeyPoint> keypoints_new;
	cv::Mat descriptors_new;

	VO.extractORBFeatures(image_left, keypoints_new, descriptors_new);

    KeyFrame refFrame = VO.getReferenceFrame();
    cv::Mat disparity_map = VO.getDisparityMap(image_left, image_right);

    if (refFrame.keypoints.empty()){
        VO.setReferenceFrame(image_left, disparity_map, keypoints_new, descriptors_new);
		return;
	}

	std::vector<cv::DMatch> matches = VO.findGoodORBFeatureMatches(keypoints_new, descriptors_new);

	// Draw top matches
    cv::Mat imMatches;
    cv::drawMatches(refFrame.image, refFrame.keypoints, image_left, keypoints_new, matches, imMatches);
    cv::imshow("Matches", imMatches);

    VO.estimatePose3D2D(keypoints_new, matches);
    VO.setReferenceFrame(image_left, disparity_map, keypoints_new, descriptors_new);
}

void VisualSLAM::runBackEndRoutine(){
	// TODO
}

void VisualSLAM::update(){
	//TODO
}
