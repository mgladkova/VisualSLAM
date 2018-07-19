#include "VisualSLAM.h"

VisualSLAM::VisualSLAM() {
    K = Eigen::Matrix3d::Identity();
}

Sophus::SE3d VisualSLAM::getPose_left(int index) {
    if (index < 0 || index >= map_left.getCumPoses().size()){
        throw std::runtime_error("VisualSLAM::getPose() : Index out of bounds");
    }

    return map_left.getCumPoses().at(index);
}

Sophus::SE3d VisualSLAM::getPose_right(int index) {
    if (index < 0 || index >= map_right.getCumPoses().size()){
        throw std::runtime_error("VisualSLAM::getPose() : Index out of bounds");
    }

    return map_right.getCumPoses().at(index);
}

Sophus::SE3d VisualSLAM::getGTPose(int index) const{
    if (index < 0 || index >= groundTruthData.size()){
        throw std::runtime_error("getGTPose() : Index out of bounds");
    }
    return groundTruthData[index];
}

std::vector<Sophus::SE3d> VisualSLAM::getPoses_left() const{
    return map_left.getCumPoses();
}

int VisualSLAM::getNumberPoses_left() const{
    return map_left.getCumPoses().size();
}

std::vector<Sophus::SE3d> VisualSLAM::getPoses_right() const{
    return map_right.getCumPoses();
}

int VisualSLAM::getNumberPoses_right() const{
    return map_right.getCumPoses().size();
}

Eigen::Matrix3d VisualSLAM::getCameraMatrix() const {
    return K;
}

double VisualSLAM::getFocalLength() const {
    return (K(0,0) + K(1,1)) / 2.0;
}

std::vector<cv::Point3f> VisualSLAM::getStructure3D_left() const{
    return map_left.getStructure3D();
}

std::vector<cv::Point3f> VisualSLAM::getStructure3D_right() const{
    return map_right.getStructure3D();
}

std::vector<std::pair<int, cv::Point2f>> VisualSLAM::getObservationsForLeftCamera(int cameraIndex){
    std::map<int, std::vector<std::pair<int, cv::Point2f>>> observations = map_left.getObservations();

    if (cameraIndex >= observations.size() || cameraIndex < 0){
        throw std::runtime_error("getObservationsForCamera() : Index out of bounds!");
    }

    return observations[cameraIndex];
}

std::vector<std::pair<int, cv::Point2f>> VisualSLAM::getObservationsForRightCamera(int cameraIndex){
    std::map<int, std::vector<std::pair<int, cv::Point2f>>> observations = map_right.getObservations();

    if (cameraIndex >= observations.size() || cameraIndex < 0){
        throw std::runtime_error("getObservationsForCamera() : Index out of bounds!");
    }

    return observations[cameraIndex];
}


cv::Rect VisualSLAM::computeROIDisparityMap(cv::Size2i src_sz, cv::Ptr<cv::stereo::StereoBinarySGBM> matcher_instance)
{
    int min_disparity = matcher_instance->getMinDisparity();
    int num_disparities = matcher_instance->getNumDisparities();
    int block_size = matcher_instance->getBlockSize();

    int bs2 = block_size/2;
    int minD = min_disparity, maxD = min_disparity + num_disparities - 1;

    int xmin = maxD + bs2;
    int xmax = src_sz.width + minD - bs2;
    int ymin = bs2;
    int ymax = src_sz.height - bs2;

    cv::Rect r(xmin, ymin, xmax - xmin, ymax - ymin);
    return r;
}

cv::Mat VisualSLAM::getDisparityMap(const cv::Mat image_left, const cv::Mat image_right){
    cv::Mat disparity;
    cv::Mat true_dmap, disparity_norm;
    cv::Rect ROI;
    int min_disparity = 0;
    int number_of_disparities = 16*6 - min_disparity;
    int kernel_size = 7;

    cv::Ptr<cv::stereo::StereoBinarySGBM> sgbm = cv::stereo::StereoBinarySGBM::create(min_disparity, number_of_disparities, kernel_size);
    // setting the penalties for sgbm
    sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);

    sgbm->setP1(8*std::pow(kernel_size, 2));
    sgbm->setP2(32*std::pow(kernel_size, 2));
    sgbm->setMinDisparity(min_disparity);
    sgbm->setUniquenessRatio(3);
    sgbm->setSpeckleWindowSize(200);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setSpekleRemovalTechnique(cv::stereo::CV_SPECKLE_REMOVAL_AVG_ALGORITHM);
    sgbm->setSubPixelInterpolationMethod(cv::stereo::CV_SIMETRICV_INTERPOLATION);

    // setting the penalties for sgbm
    ROI = computeROIDisparityMap(image_left.size(),sgbm);
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
    wls_filter = cv::ximgproc::createDisparityWLSFilterGeneric(false);
    wls_filter->setDepthDiscontinuityRadius(2);

    sgbm->compute(image_left, image_right, disparity);
    wls_filter->setLambda(8000.0);
    wls_filter->setSigmaColor(1.5);
    wls_filter->filter(disparity,image_left,disparity_norm,cv::Mat(), ROI);

    cv::Mat filtered_disp_vis;
    cv::ximgproc::getDisparityVis(disparity_norm,filtered_disp_vis,1);
    /*
    cv::namedWindow("filtered disparity", cv::WINDOW_AUTOSIZE);
    cv::imshow("filtered disparity", filtered_disp_vis);
    cv::waitKey();
    */
    filtered_disp_vis.convertTo(true_dmap, CV_32F, 1.0, 0.0);
    return true_dmap;
}

void VisualSLAM::getDataFromImageLeftForDrawing(int& cameraIndex, Sophus::SE3d& camera,
                                   std::vector<cv::Point3f>& structure3d, std::vector<int>& obsIndices,
                                   Sophus::SE3d& gtCamera){
    map_left.getDataForDrawing(cameraIndex, camera, structure3d, obsIndices, gtCamera);
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

Sophus::SE3d VisualSLAM::performFrontEndStep(cv::Mat image_left, cv::Mat disparityCurrentFrame, std::vector<cv::KeyPoint>& keyPointsPrevFrame, cv::Mat& descriptorsPrevFrame){
    cv::Mat descriptorsCurrentFrame;
    std::vector<cv::KeyPoint> keyPointsCurrentFrame;

    VO.extractORBFeatures(image_left, keyPointsCurrentFrame, descriptorsCurrentFrame);

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

Sophus::SE3d VisualSLAM::performFrontEndStepWithTracking(cv::Mat image, cv::Mat disparity_map, std::vector<cv::Point2f>& pointsCurrentFrame, std::vector<cv::Point2f>& pointsPreviousFrame, cv::Mat& prevImage, bool isLeftImage){
    int max_features = 1500;
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.03);
    cv::Size subPixWinSize(10,10);

    Sophus::SE3d pose;
    // the first image is received
    if (pointsPreviousFrame.empty() || prevImage.cols == 0){
        cv::goodFeaturesToTrack(image, pointsCurrentFrame, max_features, 0.01, 10, cv::Mat(), 3, 3, false, 0.04);
        cv::cornerSubPix(image, pointsCurrentFrame, subPixWinSize, cv::Size(-1,-1), termcrit);
        std::vector<uchar> validPoints3D_init;

        std::vector<cv::Point3f> points3D_init = VO.get3DCoordinates(pointsCurrentFrame, disparity_map, K, validPoints3D_init);
        std::vector<cv::Point3f> points3D_init_f;
        std::vector<cv::Point2f> pointsCurrenFrame_init_f;
        for (int i = 0; i < points3D_init.size(); i++){
            if (validPoints3D_init[i] && checkPoint2DCoordinates(pointsCurrentFrame[i], disparity_map)){
                pointsCurrenFrame_init_f.push_back(pointsCurrentFrame[i]);
                points3D_init_f.push_back(points3D_init[i]);
            }
        }

        std::vector<int> observedPointIndices_init(points3D_init_f.size());
        std::iota(observedPointIndices_init.begin(), observedPointIndices_init.end(), 0);

        if (isLeftImage){
            map_left.updateDataCurrentFrame(pose, pointsCurrenFrame_init_f, observedPointIndices_init, points3D_init_f, true);
        } else {
            map_right.updateDataCurrentFrame(pose, pointsCurrenFrame_init_f, observedPointIndices_init, points3D_init_f, true);
        }


        image.copyTo(prevImage);
        pointsPreviousFrame = pointsCurrenFrame_init_f;
        pointsCurrentFrame.clear();
        pointsCurrentFrame = pointsCurrenFrame_init_f;

        return pose;
    }

    int thresholdNumberFeatures = 100;
    bool init = false;
    std::vector<uchar> validPoints3D;

    std::vector<uchar> status = VO.trackFeatures(prevImage, image, pointsPreviousFrame, pointsCurrentFrame, thresholdNumberFeatures, init);
    std::vector<cv::Point3f> points3DCurrentFrame = VO.get3DCoordinates(pointsCurrentFrame, disparity_map, K, validPoints3D);

    std::vector<cv::Point2f> trackedPrevFramePoints, trackedCurrFramePoints;
    std::vector<cv::Point3f> trackedPoints3DCurrentFrame;
    std::vector<int> trackedPointIndices;

    for (int i = 0; i < status.size(); i++){
        if (status[i] && validPoints3D[i] && checkPoint2DCoordinates(pointsCurrentFrame[i], image)){
            trackedPrevFramePoints.push_back(pointsPreviousFrame[i]);
            trackedCurrFramePoints.push_back(pointsCurrentFrame[i]);
            trackedPointIndices.push_back(i);
            trackedPoints3DCurrentFrame.push_back(points3DCurrentFrame[i]);
        }
    }
    //VO.estimatePose2D2D(pointsPreviousFrame, pointsCurrentFrame, K, pose);
    VO.estimatePose3D2D(trackedPoints3DCurrentFrame, trackedPrevFramePoints, trackedCurrFramePoints,  trackedPointIndices, K, pose);

    int keyFrameStep = 1;
    int numKeyFrames = 10;

    if (init){
        std::cout << "REINITIALIZATION" << std::endl;
        std::vector<cv::Point2f> pointsCurrentFrame_reinit;
        cv::goodFeaturesToTrack(image, pointsCurrentFrame_reinit, max_features, 0.01, 10, cv::Mat(), 3, 3, 0, 0.04);
        cv::cornerSubPix(image, pointsCurrentFrame_reinit, subPixWinSize, cv::Size(-1,-1), termcrit);

        std::vector<uchar> validPoints3D_reinit;

        std::vector<cv::Point3f> points3D_reinit = VO.get3DCoordinates(pointsCurrentFrame_reinit, disparity_map, K, validPoints3D_reinit);
        std::vector<cv::Point3f> points3D_reinit_f;
        std::vector<cv::Point2f> pointsCurrentFrame_reinit_f;

        for (int i = 0; i < points3D_reinit.size(); i++){
            if (validPoints3D_reinit[i] && checkPoint2DCoordinates(pointsCurrentFrame_reinit[i], disparity_map)){
                points3D_reinit_f.push_back(points3D_reinit[i]);
                pointsCurrentFrame_reinit_f.push_back(pointsCurrentFrame_reinit[i]);
            }
        }

        std::vector<int> indicesReinit(pointsCurrentFrame_reinit_f.size());
        std::iota(indicesReinit.begin(), indicesReinit.end(), 0);

        //trackedCurrFramePoints.insert(trackedCurrFramePoints.end(), pointsCurrentFrame_reinit.begin(), pointsCurrentFrame_reinit.end());
        //points3DCurrentFrame.insert(points3DCurrentFrame.end(), points3D_reinit.begin(), points3D_reinit.end());
        //trackedPointIndices.insert(trackedPointIndices.end(), indicesReinit.begin(),indicesReinit.end());

        if (isLeftImage){
            map_left.updateDataCurrentFrame(pose, pointsCurrentFrame_reinit_f, indicesReinit, points3D_reinit_f, true);
        } else {
            map_right.updateDataCurrentFrame(pose, pointsCurrentFrame_reinit_f, indicesReinit, points3D_reinit_f, true);
        }

        pointsPreviousFrame.clear();
        pointsPreviousFrame = pointsCurrentFrame_reinit_f;
        pointsCurrentFrame.clear();
        pointsCurrentFrame = pointsCurrentFrame_reinit_f;

        image.copyTo(prevImage);
        return pose;
    }

    if (isLeftImage){
        map_left.updateDataCurrentFrame(pose, trackedCurrFramePoints, trackedPointIndices, trackedPoints3DCurrentFrame, false);

        //if ((map_left.getCurrentCameraIndex() - numKeyFrames*keyFrameStep) >= 0){
        if ((map_left.getCurrentCameraIndex() % (numKeyFrames*keyFrameStep)) == 0 && map_left.getCurrentCameraIndex() > 0){
            std::string fileName = "BAFile" + std::to_string(map_left.getCurrentCameraIndex() / (numKeyFrames*keyFrameStep)) + "L.txt";
            map_left.writeBAFile(fileName, keyFrameStep, numKeyFrames);
            BA.performBAWithKeyFrames(map_left, keyFrameStep, numKeyFrames);
        }
    } else {
        map_right.updateDataCurrentFrame(pose, trackedCurrFramePoints, trackedPointIndices, trackedPoints3DCurrentFrame, false);

        //if ((map_right.getCurrentCameraIndex() - numKeyFrames*keyFrameStep) >= 0){
        if ((map_right.getCurrentCameraIndex() % (numKeyFrames*keyFrameStep)) == 0 && map_right.getCurrentCameraIndex() > 0){
            std::string fileName = "BAFile" + std::to_string(map_right.getCurrentCameraIndex() / (numKeyFrames*keyFrameStep)) + "R.txt";
            map_right.writeBAFile(fileName, keyFrameStep, numKeyFrames);
            BA.performBAWithKeyFrames(map_right, keyFrameStep, numKeyFrames);
        }
    }

    pointsPreviousFrame.clear();
    pointsPreviousFrame = pointsCurrentFrame;

    image.copyTo(prevImage);
    return pose;
}

bool VisualSLAM::performPoseGraphOptimization(int keyFrameStep, int numKeyFrames){
    if (map_left.getCurrentCameraIndex() != map_right.getCurrentCameraIndex()){
        throw std::runtime_error("VisualSLAM::performPoseGraphOptimization() : cannot perform pose graph on non-synchronized stereo data!");
    }
    return BA.performPoseGraphOptimization(map_left, map_right, keyFrameStep, numKeyFrames);
}
