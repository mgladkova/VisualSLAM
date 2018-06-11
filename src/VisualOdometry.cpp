#include "VisualOdometry.h"
#include "BundleAdjuster.h"

//using namespace cv::xfeatures2d;
//using namespace cv;

VisualOdometry::VisualOdometry(){}

void VisualOdometry::setReferenceFrame(const cv::Mat image, const cv::Mat disparity, const std::vector<cv::KeyPoint> keypoints, const cv::Mat descriptor){
    image.copyTo(refFrame.image);
    disparity.copyTo(refFrame.disparity_map);
    refFrame.keypoints = keypoints;
    descriptor.copyTo(refFrame.descriptor);
}

KeyFrame VisualOdometry::getReferenceFrame() const {
    return refFrame;
}

void VisualOdometry::setPose(const Sophus::SE3d pose){
    this->pose = pose;
}

Sophus::SE3d VisualOdometry::getPose() const{
    return pose;
}


cv::Rect VisualOdometry::computeROIDisparityMap(cv::Size2i src_sz, cv::Ptr<cv::stereo::StereoBinarySGBM> matcher_instance)
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

cv::Mat VisualOdometry::getDisparityMap(const cv::Mat image_left, const cv::Mat image_right){
    cv::Mat disparity, true_dmap, disparity_norm;
    cv::Rect ROI;
    int min_disparity = 0;
    int number_of_disparities = 16*6 - min_disparity;
    int kernel_size = 7;

    cv::Ptr<cv::stereo::StereoBinarySGBM> sgbm = cv::stereo::StereoBinarySGBM::create(min_disparity, number_of_disparities, kernel_size);
    // setting the penalties for sgbm
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
    wls_filter->setDepthDiscontinuityRadius(1);

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
    filtered_disp_vis.convertTo(true_dmap, CV_32F, 1.0/16.0, 0.0);
    return true_dmap;
}

void VisualOdometry::extractORBFeatures(cv::Mat frame_new, std::vector<cv::KeyPoint>& keypoints_new, cv::Mat& descriptors_new){
    int max_features = 1000;
	// Detect ORB features and compute descriptors.
    cv::Ptr<cv::Feature2D> orb = cv::ORB::create(max_features);
    orb->detectAndCompute(frame_new, cv::Mat(), keypoints_new, descriptors_new);
}

std::vector<cv::DMatch> VisualOdometry::findGoodORBFeatureMatches(std::vector<cv::KeyPoint> keypoints_new, cv::Mat descriptors_new){
    const float good_match_ratio = 0.8;

    std::vector<std::vector<cv::DMatch>> vmatches;
    std::vector<cv::DMatch> matches;

    cv::Ptr<cv::DescriptorMatcher>  matcher = cv::DescriptorMatcher::create("BruteForce-Hamming(2)");
    matcher->knnMatch(refFrame.descriptor, descriptors_new, vmatches, 1);
    for (int i = 0; i < static_cast<int>(vmatches.size()); ++i) {
        if (!vmatches[i].size()) {
            continue;
        }
        matches.push_back(vmatches[i][0]);
    }
    // Sort matches by score
    std::sort(matches.begin(), matches.end());

    // Remove not so good matches
    const int numGoodMatches = matches.size() * good_match_ratio;
    matches.erase(matches.begin()+numGoodMatches, matches.end());

    return matches;
}

void VisualOdometry::get3D2DCorrespondences(std::vector<cv::KeyPoint> keypoints_new, std::vector<cv::DMatch> matches, std::vector<cv::Point3d>& p3d, std::vector<cv::Point2d>& p2d, Eigen::Matrix3d K){
    if (matches.empty()){
        throw std::runtime_error("get3d2dCorrespondences() : Input vector with keypoint matching is empty");
    }

    double b = 0.573;
    double fx = K(0,0);
    double fy = K(1,1);
    double cx = K(0,2);
    double cy = K(1,2);

    double f = (fx + fy) / 2;
    // prepare data
    for (auto &m: matches) {
        cv::Point2d p1 = refFrame.keypoints[m.queryIdx].pt;
        cv::Point2d p2 = keypoints_new[m.trainIdx].pt;
        float disparity = refFrame.disparity_map.at<float>(p1.y, p1.x);
        if (disparity){
           double z = f*b/disparity;
           double x = z*(p1.x - cx) / fx;
           double y = z*(p1.y - cy) / fy;
           //std::cout << x << " " << y << " " << z << std::endl;
           cv::Point3d p1_3d(x, y, z);
           p3d.push_back(p1_3d);
           p2d.push_back(p2);
        }
    }

    //computeAndShowPointCloud(refFrame.image, refFrame.disparity_map, b, K);

}

void VisualOdometry::get2d2dCorrespondences(std::vector<cv::KeyPoint> keypoints_new, std::vector<cv::DMatch> matches, std::vector<cv::Point2d>& p2d_1, std::vector<cv::Point2d>& p2d_2){
    if (matches.empty()){
        throw std::runtime_error("get2d2dCorrespondences() : Input vector with keypoint matching is empty");
    }

    for (auto &m: matches) {
        cv::Point2f p1 = refFrame.keypoints[m.queryIdx].pt;
        cv::Point2f p2 = keypoints_new[m.trainIdx].pt;

        p2d_1.push_back(p1);
        p2d_2.push_back(p2);
    }
}

void VisualOdometry::estimatePose3D2D(std::vector<cv::Point3d> p3d, std::vector<cv::Point2d> p2d, Eigen::Matrix3d K){
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs = cv::Mat::zeros(4,1,CV_64F);
    cv::Mat rvec,tvec,rot_matrix;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t(0,0,0);

    cv::eigen2cv(K, cameraMatrix);
    bool result=cv::solvePnPRansac(p3d,p2d,cameraMatrix, distCoeffs,rvec,tvec);

    if (result){
        cv::Rodrigues(rvec, rot_matrix);
        cv::cv2eigen(rot_matrix, R);
        cv::cv2eigen(tvec,t);
    }

    pose = Sophus::SE3d(R, t);
    std::cout << "3D-2D Pnp solved Pose: "<<std::endl<< pose.matrix() << std::endl;
}

void VisualOdometry::estimatePose2D2D(std::vector<cv::Point2d> p2d_1, std::vector<cv::Point2d> p2d_2, Eigen::Matrix3d K){
    cv::Mat tvec,rot_matrix;
    cv::Mat cameraMatrix;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t;

    cv::eigen2cv(K, cameraMatrix);

    double focal = (K(0,0) + K(1,1)) / 2;
    cv::Point2d princip_point = cv::Point2d(K(0,2), K(1,2));

    cv::Mat E = cv::findEssentialMat(p2d_1, p2d_2, focal, princip_point);
    cv::recoverPose(E, p2d_1, p2d_2, cameraMatrix, rot_matrix, tvec);

    cv::cv2eigen(rot_matrix, R);
    cv::cv2eigen(tvec,t);

    pose = Sophus::SE3d(R, t);

    std::cout << pose.matrix() << std::endl;
}

void VisualOdometry::trackFeatures(){
	// TODO
}

void VisualOdometry::computeAndShowPointCloud(const cv::Mat image_left, const cv::Mat disparity, const float baseline, Eigen::Matrix3d K) {
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> pointcloud;
    double fx = K(0,0);
    double fy = K(1,1);
    double cx = K(0,2);
    double cy = K(1,2);

    // TODO Compute point cloud using disparity
    // NOTE if your computer is slow, change v++ and u++ to v++2 and u+=2 to generate a sparser point cloud
    for (int v = 0; v < image_left.rows; v++)
        for (int u = 0; u < image_left.cols; u++) {
            /// start your code here (~6 lines)

            double z = fx*baseline/(disparity.at<float>(v,u));
            double x = (u - cx)*z / fx;
            double y = (v - cy)*z / fy;

            Eigen::Vector4d point(x, y, z,
                           image_left.at<uchar>(v, u) / 255.0); // first three components are XYZ and the last is color
            pointcloud.push_back(point);
            /// end your code here
        }

    // draw the point cloud

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3f(p[3], p[3], p[3]);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}
