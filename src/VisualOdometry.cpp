#include "VisualOdometry.h"

//TO DO harrisDection
//TO DO featureTracking
std::vector<uchar> VisualOdometry::corr2DPointsFromPreFrame2DPoints(cv::Mat previousImage, cv::Mat currImage,
                                                                            std::vector<cv::Point2f> &previousFrame2DPoints_,
                                                                            std::vector<cv::Point2f> &currFrame2DPoints) {
    // Parameters for lucas kanade optical flow
//    std::vector<cv::Point2f> currFrame2DPoints= previousFrame2DPoints_;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::Size winSize = cv::Size(31, 31);
    int maxLevel = 3;
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.03);

    cv::calcOpticalFlowPyrLK(previousImage, currImage, previousFrame2DPoints_,currFrame2DPoints,status,err,winSize,maxLevel,termcrit,0.001);
    std::cout<< "currFrame2DPoints size : "<< currFrame2DPoints.size()<<std::endl;
    // trackedCurrFrame2DPoints

//    std::cout<<" trackedCurrFrame2DPoints size "<<trackedCurrFrame2DPoints.size() << std::endl;
    int numTrackedFeactures(0);
    for (int i = 0; i < status.size() ; ++i) {
        if (status[i] == 1){
            numTrackedFeactures ++;
        }
    }
    if (numTrackedFeactures < thresholdFeactures){
        reInitial = true ;
    }


    return status;
}

bool VisualOdometry::getReInital() {
    return reInitial ;
}

cv::Rect VisualOdometry::computeROIDisparityMap(cv::Size2i src_sz,
                                                cv::Ptr<cv::stereo::StereoBinarySGBM> matcher_instance) {
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

void VisualOdometry::generateDisparityMap(const cv::Mat image_left, const cv::Mat image_right) {
    cv::Mat disparity , true_dmap, disparity_norm ;
    cv::Rect ROI ;
    int min_disparity = 0;
    int number_of_disparities = 16*6 -min_disparity;
    int kernel_size =7 ;

    cv::Ptr<cv::stereo::StereoBinarySGBM> sgbm = cv::stereo::StereoBinarySGBM::create(min_disparity, number_of_disparities, kernel_size);
    // setting the penalties for sgbm

    sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
    sgbm->setP1(8*std::pow(kernel_size,2));
    sgbm->setP2(32*std::pow(kernel_size,2));
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
    disparityMap = true_dmap ;
}
//TO DO get3DPoints
std::vector<cv::Point3f> VisualOdometry::getDepth3DPointsFromCurrImage(std::vector<cv::Point2f> &currFrame2DPoints,
                                                                        Eigen::Matrix3d K) {
    float fx = K(0,0);
    float fy = K(1,1);
    float cx = K(0,2);
    float cy = K(1,2);
//    std::cout<<"fx is : "<<fx<<" fy is : "<<fy<<" cx is : "<<cx<<" cy is : "<<cy<<std::endl;
    cv::Mat mask =disparityMap >0 ;
    double minValue;
    cv::minMaxLoc(disparityMap,&minValue,NULL,NULL,NULL,mask);
    std::vector<cv::Point3f> Points;
    for (int i = 0; i < currFrame2DPoints.size() ; ++i) {

        cv::Point2f point2D = currFrame2DPoints[i];
        float disparity = disparityMap.at<float>(point2D.y, point2D.x);
        if (disparity < 0.1)
        {
            float neighborsDisparity[4];
            if (point2D.x - 1 > 0){
                neighborsDisparity[0] = disparityMap.at<float>(point2D.y , point2D.x - 1);
            }
            if (point2D.x + 1 < disparityMap.cols ) {
                neighborsDisparity[1] = disparityMap.at<float>(point2D.y , point2D.x + 1);
            }
            if (point2D.y -1 > 0){
                neighborsDisparity[2] = disparityMap.at<float>(point2D.y-1 , point2D.x );

            }
            if (point2D.y + 1 < disparityMap.rows) {
                neighborsDisparity[3] = disparityMap.at<float>(point2D.y+1 , point2D.x );
            }
            disparity = (neighborsDisparity[0] + neighborsDisparity[1] + neighborsDisparity[2] + neighborsDisparity[3] +neighborsDisparity[4])/4;
        }
        if (disparity < 0.1 ){
            disparity = minValue;
        }
        float x,y,z;
        z = fx * baseline / disparity;
        x = ( point2D.x -cx ) * z /fx;
        y = ( point2D.y -cy ) * z /fy;

        cv::Point3f Point3D(x , y , z);
        Points.push_back(Point3D);
    }


    return Points;
}


//TO DO poseEstimate2D3DPnp
Sophus::SE3d VisualOdometry::poseEstimate2D3DPNP(std::vector<cv::Point3f> &p3d, std::vector<cv::Point2f> &p2d,Eigen::Matrix3d K,Sophus::SE3d prePose ) {

    Eigen::Matrix3d R21;
    Eigen::Vector3d t21;
    cv::Mat K_cv;
    cv::eigen2cv(K,K_cv);
//    cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type); // Assuming no lens distortion
    cv::Mat dist_coeffs = cv::Mat::zeros(4,1,CV_64F);
    cv::Mat rotationVector;
    cv::Mat translationVector;
    cv::Mat R;
    std::vector<int> inliers;
    bool result = cv::solvePnPRansac(p3d,p2d,K_cv,dist_coeffs,rotationVector,translationVector, false,100,4.0,0.99,inliers);
    if (result){
        cv::Rodrigues(rotationVector,R);
        cv::cv2eigen(R,R21);
        cv::cv2eigen(translationVector,t21);
    }

    Sophus::SE3d posePnp(R21,t21);
//    std::cout<<"pose inverse: "<<std::endl<<posePnp.inverse().matrix()<<std::endl;
//    historyPose.push_back(posePnp.inverse());

    std::cout<<"pose norm: "<<std::endl<<posePnp.log().norm() <<std::endl;
    int MAXPoseNorm = 1;
    if (posePnp.log().norm() > MAXPoseNorm){
        posePnp = prePose;
    }

    return posePnp;

}


//// visualization
//void VisualOdometry::plotTrajectoryNextStep(cv::Mat& window, int index, Eigen::Vector3d& translGTAccumulated, Eigen::Vector3d& translEstimAccumulated,
//                                            Sophus::SE3d groundTruthPose, Sophus::SE3d groundTruthPrevPose, Eigen::Matrix3d& cumR, Sophus::SE3d estimPose,  Sophus::SE3d estimPrevPose){
//    int offsetX = 300;
//    int offsetY = 300;
//
//    Sophus::SE3d pose = estimPose.inverse();
//    Sophus::SE3d prevPose = estimPrevPose.inverse();
//
//    if (index == 0){
//        translGTAccumulated = groundTruthPose.translation();
//        translEstimAccumulated = pose.translation();
//    } else {
//        translGTAccumulated = translGTAccumulated + (groundTruthPose.so3().inverse()*groundTruthPrevPose.so3())*(groundTruthPose.translation() - groundTruthPrevPose.translation());
//        translEstimAccumulated = translGTAccumulated + (pose.so3().inverse()*groundTruthPrevPose.so3())*(pose.translation() - prevPose.translation());
//    }
//    cv::circle(window, cv::Point2d(offsetX + translGTAccumulated[0], offsetY + translGTAccumulated[2]), 3, cv::Scalar(0,0,255), -1);
//    cv::circle(window, cv::Point2f(offsetX + translEstimAccumulated[0], offsetY + translEstimAccumulated[2]), 3, cv::Scalar(0,255,0), -1);
//    cv::imshow("Trajectory", window);
//    cv::waitKey(3);
//    cumR = cumR*pose.so3().matrix();
//}