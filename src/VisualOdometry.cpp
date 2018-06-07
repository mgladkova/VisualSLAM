#include "VisualOdometry.h"
//#include "BundleAdjuster.h"

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

void VisualOdometry::Motion_BA(std::vector<cv::Point3d> p3d,std::vector<cv::Point2d> p2d,Eigen::Matrix3d K,Sophus::SE3 pose,int iteration_times){
        /*  Motion BA  TEST*/

//       Eigen::Matrix3d K;
//       K<<cam.fx, 0, cam.cx, 0, cam.fy, cam.cy, 0, 0, 1;
//       std::cout<<K<<std::endl;
       int iterations = iteration_times;
       // std::cout<<"error here"<<std::endl;
       assert(p3d.size() == p2d.size());
       std::cout<<"error here "<<p3d.size()<<" "<<p2d.size()<<std::endl;
       double cost = 0, lastCost = 0;
       int nPoints = p3d.size();
       Sophus::SE3 T_esti; // estimated pose
       T_esti=pose;
       for (int iter = 0; iter < iterations; iter++) {

        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        typedef Eigen::Matrix<double, 6, 1> Vector6d;
        Vector6d b = Vector6d::Zero();
        std::cout<<"the T_esti: "<<T_esti.matrix()<<std::endl;
        cost = 0;
        // compute cost
        for (int i = 0; i < nPoints; i++) {
            // compute cost for p3d[I] and p2d[I]
            Eigen::Vector2d e(0,0);    // error
            Eigen::Vector3d p_estimate;
            Eigen::Matrix3d R ;
            Eigen::Vector3d t;
            // cout<<T_esti.matrix()<<endl;
            Eigen::Matrix4d T=T_esti.matrix();

            R << T(0,0),T(0,1),T(0,2),
                 T(1,0),T(1,1),T(1,2),
                 T(2,0),T(2,1),T(2,2);
            t << T(0,3),T(1,3),T(2,3);
            // std::cout<<" T is : "<<std::endl<<T<<std::endl;
            // std::cout<<"p3d: "<<(double)p3d[i].x<<std::endl;
            // cout<<"R is :: "<<endl<<R<<endl;
            // cout<<"t is::"<<endl<<t<<endl;
            Eigen::Vector3d P3D;
            Eigen::Vector2d P2D;
            P3D<<(double)p3d[i].x,(double)p3d[i].y,(double)p3d[i].z;
            P2D<<(double)p2d[i].x,(double)p2d[i].y;
            Eigen::Vector3d P_after_move =R*P3D+t;//R*p3d[i] +t;
            p_estimate=K*P_after_move*1./P_after_move[2];//p3d[i][2];
            e[0]=P2D[0]-p_estimate[0];
            e[1]=P2D[1]-p_estimate[1];
            // e[0]=p2d[i][0]-p_estimate[0];
            // e[1]=p2d[i][1]-p_estimate[1];
            // std::cout<<"e : "<<std::endl<<e<<std::endl;
            // compute jacobian
            Eigen::Matrix<double, 2, 6> J;
            double x(P_after_move(0)),y(P_after_move(1)),z(P_after_move(2));
            double zz=z*z;
            double yy=y*y;
            double xx=x*x;
            double xy=x*y;
            double fx=cam.fx, fy=cam.fy, cx=cam.fx,cy=cam.cy;

            J <<  fx*1./z ,      0   ,  -fx*x*1./zz , -fx*xy*1./zz     , fx+(fx*xx*1./zz), -fx*y*1./z,
                  0       ,  fy*1./z ,  -fy*y*1./zz , -fy-(fy*yy)*1./zz,  fy*x*y*1./zz   ,  fy*x*1./z;
            J=-1*J;
            // cout<<"J: "<<endl<<J<<endl;
            H += J.transpose() * J;
            b += -J.transpose() * e;

            // cout<<"error_norm is : "<<error_norm<<endl;
            cost = cost + 1./2 *(e[0]*e[0]+e[1]*e[1]);
            // cout<<"cost on line is : "<<cost<<endl;
        }
        std::cout<<"cost is :"<<cost<<std::endl;
        // while(1) {}
        // solve dx
        Vector6d dx;

        // START YOUR CODE HERE
        dx=H.ldlt().solve(b);    //the caculation of H*dx=b is right
        // cout<<"dx: "<<endl<<dx<<endl;
        // cout<<"H:"  <<endl<<H<<endl;
        // cout<<"H*dx: "<<endl<<H*dx<<endl;
        // cout<<"J^T *error =b :"<<endl<<b<<endl;
        // while(1) {}
        // END YOUR CODE HERE

        if (std::isnan(dx[0])) {
            std::cout << "result is nan!" << std::endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            // cost increase, update is not good
            std::cout << "cost: " << cost << ", last cost: " << lastCost << std::endl;
            break;
        }

        // update your estimation
        // START YOUR CODE HERE
        T_esti=Sophus::SE3::exp(dx)*T_esti;
        // cout<<"the new T_esti: "<<T_esti<<endl;
        // END YOUR CODE HERE

        lastCost = cost;

        // std::cout << "iteration " << iter << " cost=" << std::cout.precision(12) << cost << std::endl;
        // std::cout << "estimated pose: \n" << T_esti.matrix() << std::endl;

        }

        std::cout << "estimated pose: \n" << T_esti.matrix() << std::endl;
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
    std::cout << "Pose: "<<std::endl<< pose.matrix() << std::endl;


    Eigen::Matrix3d K_;
    K_<<cam.fx, 0, cam.cx, 0, cam.fy, cam.cy, 0, 0, 1;
    int iteration_times=10;
//    BundleAdjuster motion_ba;
//    motion_ba.Motion_BA(p3d,p2d,K_,pose,iteration_times);
    Motion_BA(p3d,p2d,K_,pose,iteration_times);
//    /*  Motion BA  TEST*/

//   Eigen::Matrix3d K;
//   K<<cam.fx, 0, cam.cx, 0, cam.fy, cam.cy, 0, 0, 1;
//   std::cout<<K<<std::endl;
//   int iterations = 10;
//   // std::cout<<"error here"<<std::endl;
//   assert(p3d.size() == p2d.size());
//   std::cout<<"error here "<<p3d.size()<<" "<<p2d.size()<<std::endl;
//   double cost = 0, lastCost = 0;
//   int nPoints = p3d.size();
//   Sophus::SE3 T_esti; // estimated pose
//   T_esti=pose;
//   for (int iter = 0; iter < iterations; iter++) {

//    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
//    typedef Eigen::Matrix<double, 6, 1> Vector6d;
//    Vector6d b = Vector6d::Zero();
//    std::cout<<"the T_esti: "<<T_esti.matrix()<<std::endl;
//    cost = 0;
//    // compute cost
//    for (int i = 0; i < nPoints; i++) {
//        // compute cost for p3d[I] and p2d[I]
//        Eigen::Vector2d e(0,0);    // error
//        Eigen::Vector3d p_estimate;
//        Eigen::Matrix3d R ;
//        Eigen::Vector3d t;
//        // cout<<T_esti.matrix()<<endl;
//        Eigen::Matrix4d T=T_esti.matrix();

//        R << T(0,0),T(0,1),T(0,2),
//             T(1,0),T(1,1),T(1,2),
//             T(2,0),T(2,1),T(2,2);
//        t << T(0,3),T(1,3),T(2,3);
//        // std::cout<<" T is : "<<std::endl<<T<<std::endl;
//        // std::cout<<"p3d: "<<(double)p3d[i].x<<std::endl;
//        // cout<<"R is :: "<<endl<<R<<endl;
//        // cout<<"t is::"<<endl<<t<<endl;
//        Eigen::Vector3d P3D;
//        Eigen::Vector2d P2D;
//        P3D<<(double)p3d[i].x,(double)p3d[i].y,(double)p3d[i].z;
//        P2D<<(double)p2d[i].x,(double)p2d[i].y;
//        Eigen::Vector3d P_after_move =R*P3D+t;//R*p3d[i] +t;
//        p_estimate=K*P_after_move*1./P_after_move[2];//p3d[i][2];
//        e[0]=P2D[0]-p_estimate[0];
//        e[1]=P2D[1]-p_estimate[1];
//        // e[0]=p2d[i][0]-p_estimate[0];
//        // e[1]=p2d[i][1]-p_estimate[1];
//        // std::cout<<"e : "<<std::endl<<e<<std::endl;
//        // compute jacobian
//        Eigen::Matrix<double, 2, 6> J;
//        double x(P_after_move(0)),y(P_after_move(1)),z(P_after_move(2));
//        double zz=z*z;
//        double yy=y*y;
//        double xx=x*x;
//        double xy=x*y;
//        double fx=cam.fx, fy=cam.fy, cx=cam.fx,cy=cam.cy;
        
//        J <<  fx*1./z ,      0   ,  -fx*x*1./zz , -fx*xy*1./zz     , fx+(fx*xx*1./zz), -fx*y*1./z,
//              0       ,  fy*1./z ,  -fy*y*1./zz , -fy-(fy*yy)*1./zz,  fy*x*y*1./zz   ,  fy*x*1./z;
//        J=-1*J;
//        // cout<<"J: "<<endl<<J<<endl;
//        H += J.transpose() * J;
//        b += -J.transpose() * e;

//        // cout<<"error_norm is : "<<error_norm<<endl;
//        cost = cost + 1./2 *(e[0]*e[0]+e[1]*e[1]);
//        // cout<<"cost on line is : "<<cost<<endl;
//    }
//    std::cout<<"cost is :"<<cost<<std::endl;
//    // while(1) {}
//    // solve dx
//    Vector6d dx;

//    // START YOUR CODE HERE
//    dx=H.ldlt().solve(b);    //the caculation of H*dx=b is right
//    // cout<<"dx: "<<endl<<dx<<endl;
//    // cout<<"H:"  <<endl<<H<<endl;
//    // cout<<"H*dx: "<<endl<<H*dx<<endl;
//    // cout<<"J^T *error =b :"<<endl<<b<<endl;
//    // while(1) {}
//    // END YOUR CODE HERE

//    if (std::isnan(dx[0])) {
//        std::cout << "result is nan!" << std::endl;
//        break;
//    }

//    if (iter > 0 && cost >= lastCost) {
//        // cost increase, update is not good
//        std::cout << "cost: " << cost << ", last cost: " << lastCost << std::endl;
//        break;
//    }

//    // update your estimation
//    // START YOUR CODE HERE
//    T_esti=Sophus::SE3::exp(dx)*T_esti;
//    // cout<<"the new T_esti: "<<T_esti<<endl;
//    // END YOUR CODE HERE

//    lastCost = cost;

//    // std::cout << "iteration " << iter << " cost=" << std::cout.precision(12) << cost << std::endl;
//    // std::cout << "estimated pose: \n" << T_esti.matrix() << std::endl;

//    }

//    std::cout << "estimated pose: \n" << T_esti.matrix() << std::endl;

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
