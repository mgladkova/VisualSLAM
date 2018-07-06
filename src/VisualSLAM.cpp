#include "VisualSLAM.h"

VisualSLAM::VisualSLAM() {
    K = Eigen::Matrix3d::Identity();
}

Sophus::SE3d VisualSLAM::getPose(int index) {
    if (index < 0 || index >= map.getCumPoses().size()){
        throw std::runtime_error("VisualSLAM::getPose() : Index out of bounds");
    }

    return map.getCumPoses().at(index);
}

Sophus::SE3d VisualSLAM::getGTPose(int index) const{
    if (index < 0 || index >= groundTruthData.size()){
        throw std::runtime_error("getGTPose() : Index out of bounds");
    }
    return groundTruthData[index];
}

std::vector<Sophus::SE3d> VisualSLAM::getPoses() const{
    return map.getCumPoses();
}

int VisualSLAM::getNumberPoses() const{
    return map.getCumPoses().size();
}

Eigen::Matrix3d VisualSLAM::getCameraMatrix() const {
    return K;
}

double VisualSLAM::getFocalLength() const {
    return (K(0,0) + K(1,1)) / 2.0;
}

std::vector<cv::Point3f> VisualSLAM::getStructure3D() const{
    return map.getStructure3D();
}

std::vector<std::pair<int, cv::Point2f>> VisualSLAM::getObservationsForCamera(int cameraIndex){
    std::map<int, std::vector<std::pair<int, cv::Point2f>>> observations = map.getObservations();

    if (cameraIndex >= observations.size() || cameraIndex < 0){
        throw std::runtime_error("getObservationsForCamera() : Index out of bounds!");
    }

    return observations[cameraIndex];
}

bool VisualSLAM::checkPoint2DCoordinates(cv::Point2f point, cv::Mat image){
    return point.x >= 0 && point.x < image.cols && point.y < image.rows && point.y >= 0;
}

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
    K(0,0) = data[0];
    K(1,1) = data[5];
    K(0,2) = data[2];
    K(1,2) = data[6];

}

void VisualSLAM::readGroundTruthData(std::string fileName, int numberFrames, std::vector<Sophus::SE3d>& groundTruthData){
    std::ifstream inFile;
    inFile.open(fileName, std::ifstream::in);

    if (!inFile){
        throw std::runtime_error("readGroundTruthData() : Cannot read the file with ground truth data");
    }
    if (numberFrames <= 0){
        throw std::runtime_error("readGroundTruthData() : Number of frames is non-positive!");
    }

    groundTruthData.clear();

    int i = 0;
    while(i < numberFrames && !inFile.eof()){
        double rotationElements[9], translationElements[3];
        int k = 0;
        for (int j = 1; j <= 12; j++){
            if (j % 4 == 0){
                inFile >> translationElements[j / 4 - 1];
            } else {
                inFile >> rotationElements[k++];
            }
        }
        cv::Mat R_CV = cv::Mat(3,3, CV_64F, rotationElements);
        Eigen::Matrix3d R_Eigen;
        cv::cv2eigen(R_CV, R_Eigen);
        Sophus::SE3d newPose = Sophus::SE3d(Eigen::Quaterniond(R_Eigen), Eigen::Vector3d(translationElements));
        groundTruthData.push_back(newPose);
        i++;
    }

    this->groundTruthData = groundTruthData;
}

Sophus::SE3d VisualSLAM::performFrontEndStep(cv::Mat image_left, cv::Mat image_right, std::vector<cv::KeyPoint>& keyPointsPrevFrame, cv::Mat& descriptorsPrevFrame){
    cv::Mat descriptorsCurrentFrame;
    std::vector<cv::KeyPoint> keyPointsCurrentFrame;

    VO.extractORBFeatures(image_left, keyPointsCurrentFrame, descriptorsCurrentFrame);
    cv::Mat disparityCurrentFrame = VO.getDisparityMap(image_left, image_right);

    Sophus::SE3d pose;
    if (keyPointsPrevFrame.empty()){
      keyPointsPrevFrame = keyPointsCurrentFrame;
      descriptorsCurrentFrame.copyTo(descriptorsPrevFrame);

      return pose;
    }

    std::vector<cv::DMatch> matches = VO.findGoodORBFeatureMatches(keyPointsPrevFrame, keyPointsCurrentFrame, descriptorsPrevFrame, descriptorsCurrentFrame);

    std::vector<cv::Point2f> p2d_prevFrame, p2d_currFrame;

    VO.get2D2DCorrespondences(keyPointsPrevFrame, keyPointsCurrentFrame, matches, p2d_prevFrame, p2d_currFrame);

    std::vector<cv::Point3f> p3d_currFrame = VO.get3DCoordinates(p2d_currFrame, disparityCurrentFrame, K);

    std::vector<int> indices;

    VO.estimatePose3D2D(p3d_currFrame, p2d_prevFrame, p2d_currFrame, indices, K, pose);

    descriptorsCurrentFrame.copyTo(descriptorsPrevFrame);

    keyPointsPrevFrame.clear();
    keyPointsPrevFrame = keyPointsCurrentFrame;

    return pose;
}

Sophus::SE3d VisualSLAM::performFrontEndStepWithTracking(cv::Mat image_left, cv::Mat image_right, std::vector<cv::Point2f>& pointsCurrentFrame, std::vector<cv::Point2f>& pointsPreviousFrame, cv::Mat& prevImageLeft){
    int max_features = 550;
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.03);
    cv::Size subPixWinSize(10,10);

    cv::Mat disparity_map = VO.getDisparityMap(image_left, image_right);
    Sophus::SE3d pose;
    // the first image is received
    if (pointsPreviousFrame.empty()){
        cv::goodFeaturesToTrack(image_left, pointsCurrentFrame, max_features, 0.01, 10, cv::Mat(), 3, 3, false, 0.04);
        cv::cornerSubPix(image_left, pointsCurrentFrame, subPixWinSize, cv::Size(-1,-1), termcrit);

        std::vector<cv::Point3f> points3D = VO.get3DCoordinates(pointsCurrentFrame, disparity_map, K);

        int maxDistance = 100;
        for (int i = 0; i < points3D.size(); i++){
            if (points3D[i].z > maxDistance || !checkPoint2DCoordinates(pointsCurrentFrame[i], disparity_map)){
                points3D.erase(points3D.begin() + i);
                pointsCurrentFrame.erase(pointsCurrentFrame.begin() + i);
            }
        }

        std::vector<int> indices(pointsCurrentFrame.size());
        std::iota(indices.begin(), indices.end(), 0);

        map.addObservations(indices, pointsCurrentFrame, true);

        map.updateCumulativePose(pose);
        map.addPoints3D(points3D);

        map.updateCameraIndex();

        image_left.copyTo(prevImageLeft);
        pointsPreviousFrame.clear();
        pointsPreviousFrame = pointsCurrentFrame;

        return pose;
    }

    int thresholdNumberFeatures = 150;
    bool init = false;

    std::vector<uchar> status = VO.trackFeatures(prevImageLeft, image_left, pointsPreviousFrame, pointsCurrentFrame, thresholdNumberFeatures, init);
    std::vector<cv::Point2f> trackedPrevFramePoints, trackedCurrFramePoints;
    std::vector<int> indices;

    for (int i = 0; i < status.size(); i++){
        if (status[i] && checkPoint2DCoordinates(pointsCurrentFrame[i], image_left)){
            trackedPrevFramePoints.push_back(pointsPreviousFrame[i]);
            trackedCurrFramePoints.push_back(pointsCurrentFrame[i]);
            indices.push_back(i);
            //std::cout << "Point at: " << pointsCurrentFrame[i].x << " " << pointsCurrentFrame[i].y << std::endl;
        }
    }

    std::vector<cv::Point3f> points3DCurrentFrame = VO.get3DCoordinates(trackedCurrFramePoints, disparity_map, K);

    for (int i = 0; i < points3DCurrentFrame.size(); i++){
        if (points3DCurrentFrame[i].z < 0){
            std::cout << "POINTS 3D: " << points3DCurrentFrame[i].x << " " << points3DCurrentFrame[i].y << " "
                      << points3DCurrentFrame[i].z << std::endl;
        }
    }

    //VO.estimatePose2D2D(pointsPreviousFrame, pointsCurrentFrame, K, pose);
    VO.estimatePose3D2D(points3DCurrentFrame, trackedPrevFramePoints, trackedCurrFramePoints,  indices, K, pose);

    std::cout << "SIZES: " << indices.size() << " " << trackedCurrFramePoints.size() << std::endl;

    map.addObservations(indices, trackedCurrFramePoints, false);
    map.updateCumulativePose(pose);

    int keyFrameStep = 5;
    int numKeyFrames = 5;

    if (init){
        std::cout << "REINITIALIZATION" << std::endl;
        pointsCurrentFrame.clear();
        cv::goodFeaturesToTrack(image_left, pointsCurrentFrame, max_features, 0.01, 10, cv::Mat(), 3, 3, 0, 0.04);
        cv::cornerSubPix(image_left, pointsCurrentFrame, subPixWinSize, cv::Size(-1,-1), termcrit);

        std::vector<cv::Point3f> points3D = VO.get3DCoordinates(pointsCurrentFrame, disparity_map, K);

        int maxDistance = 100;
        for (int i = 0; i < points3D.size(); i++){
            if (points3D[i].z > maxDistance || !checkPoint2DCoordinates(pointsCurrentFrame[i], disparity_map)){
                points3D.erase(points3D.begin() + i);
                pointsCurrentFrame.erase(pointsCurrentFrame.begin() + i);
            }
        }

        std::vector<int> indices(pointsCurrentFrame.size());
        std::iota(indices.begin(), indices.end(), 0);

        map.addObservations(indices, pointsCurrentFrame, true);
        map.addPoints3D(points3D);
    }

    map.updateCameraIndex();

    if (map.getCurrentCameraIndex() % (numKeyFrames*keyFrameStep) == 0 && map.getCurrentCameraIndex() > 0){
    //int overlapBAPatches = std::max(1, int(keyFrameStep / 2));
    //if ((map.getCurrentCameraIndex() - numKeyFrames*keyFrameStep) >= 0 &&
    //    (map.getCurrentCameraIndex() - numKeyFrames*keyFrameStep) % overlapBAPatches == 0){
        std::string fileName = "BAFile" + std::to_string(map.getCurrentCameraIndex() / (numKeyFrames*keyFrameStep)) + ".txt";
        //map.writeBAFile(fileName, keyFrameStep, numKeyFrames);
        BA.optimizeCameraPosesForKeyframes(map, keyFrameStep, numKeyFrames);
    }

    pointsPreviousFrame.clear();
    pointsPreviousFrame = pointsCurrentFrame;

    image_left.copyTo(prevImageLeft);
    return pose;
}
