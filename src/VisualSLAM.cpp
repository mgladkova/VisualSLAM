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

void VisualSLAM::getDataForDrawing(int& cameraIndex, Sophus::SE3d& camera, std::vector<cv::Point3f>& structure3d, std::vector<int>& obsIndices, Sophus::SE3d& gtCamera){
    map.getDataForDrawing(cameraIndex, camera, structure3d, obsIndices, gtCamera);
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

    std::vector<uchar> validPoints3D;
    std::vector<cv::Point3f> p3d_currFrame = VO.get3DCoordinates(p2d_currFrame, disparityCurrentFrame, K, validPoints3D);

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
    if (pointsPreviousFrame.empty() || prevImageLeft.cols == 0){
        cv::goodFeaturesToTrack(image_left, pointsCurrentFrame, max_features, 0.01, 10, cv::Mat(), 3, 3, false, 0.04);
        cv::cornerSubPix(image_left, pointsCurrentFrame, subPixWinSize, cv::Size(-1,-1), termcrit);
        std::vector<uchar> validPoints3D_init;

        std::vector<cv::Point3f> points3D_init = VO.get3DCoordinates(pointsCurrentFrame, disparity_map, K, validPoints3D_init);
        std::vector<cv::Point3f> points3D_init_f;
        std::vector<cv::Point2f> pointsCurrenFrame_init_f;
        int k = getStructure3D().size();
        for (int i = 0; i < points3D_init.size(); i++){
            if (validPoints3D_init[i] && checkPoint2DCoordinates(pointsCurrentFrame[i], disparity_map)){
                pointsCurrenFrame_init_f.push_back(pointsCurrentFrame[i]);
                points3D_init_f.push_back(points3D_init[i]);
                std::cout << "ADDING " << k++ << " " << points3D_init[i].x << " " << points3D_init[i].y << " " << points3D_init[i].z << " " << std::endl;
            }
        }

        std::vector<int> observedPointIndices_init(points3D_init_f.size());
        std::iota(observedPointIndices_init.begin(), observedPointIndices_init.end(), 0);

        map.updateDataCurrentFrame(pose, pointsCurrenFrame_init_f, observedPointIndices_init, points3D_init_f, true);

        image_left.copyTo(prevImageLeft);
        pointsPreviousFrame = pointsCurrenFrame_init_f;
        pointsCurrentFrame.clear();
        pointsCurrentFrame = pointsCurrenFrame_init_f;

        return pose;
    }

    int thresholdNumberFeatures = 150;
    bool init = false;
    std::vector<uchar> validPoints3D;

    std::vector<uchar> status = VO.trackFeatures(prevImageLeft, image_left, pointsPreviousFrame, pointsCurrentFrame, thresholdNumberFeatures, init);
    std::vector<cv::Point3f> points3DCurrentFrame = VO.get3DCoordinates(pointsCurrentFrame, disparity_map, K, validPoints3D);

    std::vector<cv::Point2f> trackedPrevFramePoints, trackedCurrFramePoints;
    std::vector<cv::Point3f> trackedPoints3DCurrentFrame;
    std::vector<int> trackedPointIndices;

    for (int i = 0; i < status.size(); i++){
        if (status[i] && validPoints3D[i] && checkPoint2DCoordinates(pointsCurrentFrame[i], image_left)){
            trackedPrevFramePoints.push_back(pointsPreviousFrame[i]);
            trackedCurrFramePoints.push_back(pointsCurrentFrame[i]);
            trackedPointIndices.push_back(i);
            trackedPoints3DCurrentFrame.push_back(points3DCurrentFrame[i]);
        }
    }
    //VO.estimatePose2D2D(pointsPreviousFrame, pointsCurrentFrame, K, pose);
    VO.estimatePose3D2D(trackedPoints3DCurrentFrame, trackedPrevFramePoints, trackedCurrFramePoints,  trackedPointIndices, K, pose);

    int keyFrameStep = 1;
    int numKeyFrames = 5;

    if (init){
        std::cout << "REINITIALIZATION" << std::endl;
        std::vector<cv::Point2f> pointsCurrentFrame_reinit;
        cv::goodFeaturesToTrack(image_left, pointsCurrentFrame_reinit, max_features, 0.01, 10, cv::Mat(), 3, 3, 0, 0.04);
        cv::cornerSubPix(image_left, pointsCurrentFrame_reinit, subPixWinSize, cv::Size(-1,-1), termcrit);

        std::vector<uchar> validPoints3D_reinit;

        std::vector<cv::Point3f> points3D_reinit = VO.get3DCoordinates(pointsCurrentFrame_reinit, disparity_map, K, validPoints3D_reinit);
        std::vector<cv::Point3f> points3D_reinit_f;
        std::vector<cv::Point2f> pointsCurrentFrame_reinit_f;
        int k = getStructure3D().size();

        for (int i = 0; i < points3D_reinit.size(); i++){
            if (validPoints3D_reinit[i] && checkPoint2DCoordinates(pointsCurrentFrame_reinit[i], disparity_map)){
                points3D_reinit_f.push_back(points3D_reinit[i]);
                pointsCurrentFrame_reinit_f.push_back(pointsCurrentFrame_reinit[i]);
                std::cout << "ADDING " << k++ << " " << points3D_reinit[i].x << " " << points3D_reinit[i].y << " " << points3D_reinit[i].z << " " << std::endl;
            }
        }

        std::vector<int> indicesReinit(pointsCurrentFrame_reinit_f.size());
        std::iota(indicesReinit.begin(), indicesReinit.end(), 0);

        //trackedCurrFramePoints.insert(trackedCurrFramePoints.end(), pointsCurrentFrame_reinit.begin(), pointsCurrentFrame_reinit.end());
        //points3DCurrentFrame.insert(points3DCurrentFrame.end(), points3D_reinit.begin(), points3D_reinit.end());
        //trackedPointIndices.insert(trackedPointIndices.end(), indicesReinit.begin(),indicesReinit.end());

        map.updateDataCurrentFrame(pose, pointsCurrentFrame_reinit_f, indicesReinit, points3D_reinit_f, true);
        pointsPreviousFrame.clear();
        pointsPreviousFrame = pointsCurrentFrame_reinit_f;
        pointsCurrentFrame.clear();
        pointsCurrentFrame = pointsCurrentFrame_reinit_f;

        image_left.copyTo(prevImageLeft);
        return pose;
    }

    map.updateDataCurrentFrame(pose, trackedCurrFramePoints, trackedPointIndices, trackedPoints3DCurrentFrame, false);


    if ((map.getCurrentCameraIndex() - numKeyFrames*keyFrameStep) >= 0){
    //if ((map.getCurrentCameraIndex() % (numKeyFrames*keyFrameStep)) == 0 && map.getCurrentCameraIndex() > 0){
        std::string fileName = "BAFile" + std::to_string(map.getCurrentCameraIndex() / (numKeyFrames*keyFrameStep)) + ".txt";
        map.writeBAFile(fileName, keyFrameStep, numKeyFrames);
        BA.optimizeCameraPosesForKeyframes(map, keyFrameStep, numKeyFrames);
    }

    pointsPreviousFrame.clear();
    pointsPreviousFrame = pointsCurrentFrame;

    image_left.copyTo(prevImageLeft);
    return pose;
}
