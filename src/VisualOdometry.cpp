#include "VisualOdometry.h"
#include "BundleAdjuster.h"

//using namespace cv::xfeatures2d;
//using namespace cv;

VisualOdometry::VisualOdometry(){
	cam.fx = 0.0;
	cam.fy = 0.0;
	cam.cx = 0.0;
	cam.cy = 0.0;
}

void VisualOdometry::setReferenceFrame(const cv::Mat image, const cv::Mat disparity, const std::vector<cv::KeyPoint> keypoints, const cv::Mat descriptor){
    image.copyTo(refFrame.image);
    disparity.copyTo(refFrame.disparity_map);
    refFrame.keypoints = keypoints;
    descriptor.copyTo(refFrame.descriptor);
}

void VisualOdometry::setCamera(double fx, double fy, double cx, double cy){
	this->cam.fx = fx;
	this->cam.fy = fy;
	this->cam.cx = cx;
	this->cam.cy = cy;
}

Camera VisualOdometry::getCamera() const {
	return cam;
}

KeyFrame VisualOdometry::getReferenceFrame() const {
    return refFrame;
}

cv::Mat VisualOdometry::getDisparityMap(const cv::Mat image_left, const cv::Mat image_right){
    cv::Mat disparity, disparity_norm;
    int number_of_disparities = 16*8;
    int kernel_size = 7;

    cv::Ptr<cv::stereo::StereoBinarySGBM> sgbm = cv::stereo::StereoBinarySGBM::create(0, number_of_disparities, kernel_size);
    // setting the penalties for sgbm
    sgbm->setP1(10);
    sgbm->setP2(240);
    sgbm->setMinDisparity(4);
    sgbm->setUniquenessRatio(4);
    sgbm->setSpeckleWindowSize(45);
    sgbm->setSpeckleRange(2);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setBinaryKernelType(CV_8UC1);
    sgbm->setSpekleRemovalTechnique(cv::stereo::CV_SPECKLE_REMOVAL_AVG_ALGORITHM);
    sgbm->setSubPixelInterpolationMethod(cv::stereo::CV_SIMETRICV_INTERPOLATION);
    sgbm->compute(image_left, image_right, disparity);
    double minVal; double maxVal;
    minMaxLoc(disparity, &minVal, &maxVal);
    disparity.convertTo(disparity_norm, CV_8U, 255 / (maxVal - minVal));
    return disparity_norm;
}

void VisualOdometry::extractORBFeatures(cv::Mat frame_new, std::vector<cv::KeyPoint>& keypoints_new, cv::Mat& descriptors_new){
    int max_features = 400;
	// Detect ORB features and compute descriptors.
    cv::Ptr<cv::Feature2D> orb = cv::ORB::create(max_features);
    orb->detectAndCompute(frame_new, cv::Mat(), keypoints_new, descriptors_new);
}

std::vector<cv::DMatch> VisualOdometry::findGoodORBFeatureMatches(std::vector<cv::KeyPoint> keypoints_new, cv::Mat descriptors_new){
    float good_match_ratio = 0.8;
    std::vector<cv::DMatch> matches;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
    matcher->match(refFrame.descriptor, descriptors_new, matches, cv::Mat());
	// Sort matches by score
	std::sort(matches.begin(), matches.end());
	// Remove not so good matches
	const int numGoodMatches = matches.size() * good_match_ratio;
	matches.erase(matches.begin()+numGoodMatches, matches.end());
	return matches;
}

void VisualOdometry::estimatePose3D2D(std::vector<cv::KeyPoint> keypoints_new, std::vector<cv::DMatch> matches){
    std::vector<cv::Point2d> p2d;
    std::vector<cv::Point3d> p3d;
    float depth_scale = 100.f;
	// prepare data
    for (auto &m: matches) {
        cv::Point2d p1 = refFrame.keypoints[m.queryIdx].pt;
        cv::Point2d p2 = keypoints_new[m.trainIdx].pt;
        double depth = refFrame.disparity_map.at<uint8_t>(p1.y, p1.x) / depth_scale;
        if (depth){
           double x = depth*(p1.x - cam.cx) / cam.fx;
           double y = depth*(p1.y - cam.cy) / cam.fy;
           cv::Point3d p2_3d(x, y, depth);
           p3d.push_back(p2_3d);
           p2d.push_back(p1);
        }
    }
    double data[] = {cam.fx, 0, cam.cx, 0, cam.fy, cam.cy, 0, 0, 1};
    cv::Mat cameraMatrix = cv::Mat(3, 3, CV_32F, data);
    cv::Mat distCoeffs = cv::Mat::zeros(4,1,CV_32F);
    cv::Mat rvec,tvec,rot_matrix;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t;
    bool result=cv::solvePnP(p3d,p2d,cameraMatrix, distCoeffs,rvec,tvec, false, CV_EPNP);
    if (result){
        cv::Rodrigues(rvec, rot_matrix);
        cv::cv2eigen(rot_matrix, R);
        cv::cv2eigen(tvec,t);
    }
    pose = Sophus::SE3(R, t);
    std::cout << "3D-2D Pnp solved Pose: "<<std::endl<< pose.matrix() << std::endl;

    // each motion bundle adjustment (reference image's 3D point and matched image's 2D point)
    Eigen::Matrix3d K_;
    K_<<cam.fx, 0, cam.cx, 0, cam.fy, cam.cy, 0, 0, 1;
    int iteration_times=10;
    BundleAdjuster motion_ba;
    Sophus::SE3 Esti_pose=motion_ba.Motion_BA(p3d,p2d,K_,pose,iteration_times);
    Esti_pose_vector.push_back(Esti_pose);
    std::cout<<"After motion ba the Esti_pose is:"<<std::endl<<Esti_pose_vector[0].matrix()<<std::endl;

}

void VisualOdometry::estimatePose2D2D(std::vector<cv::KeyPoint> keypoints_new, std::vector<cv::DMatch> matches){
    std::vector<cv::Point2f> p2d_1, p2d_2;
    cv::Mat tvec,rot_matrix;

    for (auto &m: matches) {
        cv::Point2f p1 = refFrame.keypoints[m.queryIdx].pt;
        cv::Point2f p2 = keypoints_new[m.trainIdx].pt;

        p2d_1.push_back(p1);
        p2d_2.push_back(p2);
    }

    double data[] = {cam.fx, 0, cam.cx, 0, cam.fy, cam.cy, 0, 0, 1};
    cv::Mat cameraMatrix = cv::Mat(3, 3, CV_32F, data);
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t;

    double focal = (cam.fx + cam.fy) / 2;
    cv::Point2d princip_point = cv::Point2d(cam.cx, cam.cy);

    cv::Mat E = cv::findEssentialMat(p2d_1, p2d_2, focal, princip_point);
    cv::recoverPose(E, p2d_1, p2d_2, cameraMatrix, rot_matrix, tvec);

    cv::cv2eigen(rot_matrix, R);
    cv::cv2eigen(tvec,t);

    pose = Sophus::SE3(R.transpose(), -R.transpose()*t);
}

void VisualOdometry::trackFeatures(){
	// TODO
}
