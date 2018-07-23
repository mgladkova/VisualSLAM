#include "VisualOdometry.h"
#include "BundleAdjuster.h"

//using namespace cv::xfeatures2d;
//using namespace cv;

VisualOdometry::VisualOdometry(){}

void VisualOdometry::setReferenceFrame(const cv::Mat image, const cv::Mat disparity_map, const std::vector<cv::KeyPoint> keypoints, const cv::Mat descriptors){
    image.copyTo(refFrame.image);
    disparity_map.copyTo(refFrame.disparity_map);
    refFrame.keypoints = keypoints;
    refFrame.descriptor = descriptors;
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

void VisualOdometry::setKeyFrameKeypoints(std::vector<cv::KeyPoint> updatedKeypoints){
    refFrame.keypoints.clear();
    refFrame.keypoints = updatedKeypoints;
}

void VisualOdometry::extractORBFeatures(cv::Mat frame_new, std::vector<cv::KeyPoint>& keypoints_new, cv::Mat& descriptors_new){
    int max_features = 200;
	// Detect ORB features and compute descriptors.
    cv::Ptr<cv::Feature2D> orb = cv::ORB::create(max_features);
    orb->detect(frame_new, keypoints_new);
    orb->compute(frame_new, keypoints_new, descriptors_new);
}

std::vector<cv::DMatch> VisualOdometry::findGoodORBFeatureMatches(std::vector<cv::KeyPoint> keypointsPrevFrame, std::vector<cv::KeyPoint> keypointsCurrentFrame,
                                                                  cv::Mat descriptorsPrevFrame, cv::Mat descriptorsCurrentFrame){
    const float good_match_ratio = 0.8;
    std::vector<cv::DMatch> matches;

    cv::Ptr<cv::DescriptorMatcher>  matcher = cv::DescriptorMatcher::create("BruteForce-Hamming(2)");
    matcher->match(descriptorsPrevFrame, descriptorsCurrentFrame, matches);

    // Sort matches by score
    std::sort(matches.begin(), matches.end());
    // Remove not so good matches
    const int numGoodMatches = matches.size() * good_match_ratio;
    matches.erase(matches.begin()+numGoodMatches, matches.end());

    return matches;
}

std::vector<cv::Point2d> VisualOdometry::get2DPointsKeyFrame(){
    std::vector<cv::Point2d> points;
    for (auto& kpt : refFrame.keypoints){
        points.push_back(kpt.pt);
    }

    return points;
}

std::vector<cv::Point3f> VisualOdometry::get3DCoordinates(std::vector<cv::KeyPoint> keypoints, cv::Mat disparity_map, Eigen::Matrix3d K, std::vector<uchar>& status){
    std::vector<cv::Point2f> points2D;

    for (auto& keyPt: keypoints){
        points2D.push_back(keyPt.pt);
    }

    return get3DCoordinates(points2D, disparity_map, K, status);
}

std::vector<cv::Point3f> VisualOdometry::get3DCoordinates(std::vector<cv::Point2f> points2D, cv::Mat disparity_map, Eigen::Matrix3d K, std::vector<uchar>& status){
    float b = 0.53716;
    float fx = K(0,0);
    float fy = K(1,1);
    float cx = K(0,2);
    float cy = K(1,2);

    float f = (fx + fy) / 2;
    std::vector<cv::Point3f> points3D(points2D.size());
    status.clear();
    status.resize(points2D.size());

    float MINDISPARITY = 0.001;
    float MAXDISPARITY = 1000;

    for (int i = 0; i < points2D.size(); i++) {
        cv::Point2f p = points2D[i];
        if (p.x < 0 || p.x >= disparity_map.cols || p.y < 0 || p.y >= disparity_map.rows){
            status[i] = 0;
            continue;
        }
        //std::cout << p.x << " " << p.y << std::endl;
        float disparity = disparity_map.at<float>(p.y, p.x);
        if (disparity < MINDISPARITY || disparity > MAXDISPARITY){
            // compute the average depth over the path around the point
            float neighborDisparities[4] = {0};
            if (p.x - 1 >= 0){
                neighborDisparities[0] = disparity_map.at<float>(p.y, p.x - 1);
            }

            if (p.y - 1 >= 0){
                neighborDisparities[1] = disparity_map.at<float>(p.y - 1, p.x);
            }

            if (p.x + 1 < disparity_map.cols){
                neighborDisparities[2] = disparity_map.at<float>(p.y, p.x + 1);
            }

            if (p.y + 1 < disparity_map.rows){
                neighborDisparities[3] = disparity_map.at<float>(p.y + 1, p.x);
            }

            disparity = (neighborDisparities[0] + neighborDisparities[1] + neighborDisparities[2] + neighborDisparities[3]) / 4;
        }

        float z = f*b/disparity;
        float x = z*(p.x - cx) / fx;
        float y = z*(p.y - cy) / fy;

        if (z < MINDISPARITY || z > MAXDISPARITY){
            status[i] = 0;
        } else {
            status[i] = 1;
            //std::cout << x << " " << y << " " << z << " " << disparity << std::endl;
        }

        points3D[i] = cv::Point3f(x,y,z);
    }

    return points3D;
}

void VisualOdometry::get2D2DCorrespondences(std::vector<cv::KeyPoint> keypointsPrevFrame, std::vector<cv::KeyPoint> keypointsCurrentFrame, std::vector<cv::DMatch> matches, std::vector<cv::Point2f>& p2dPrevFrame, std::vector<cv::Point2f>& p2dCurrentFrame){
    if (matches.empty()){
        throw std::runtime_error("get2d2dCorrespondences() : Input vector with keypoint matching is empty");
    }

    for (auto &m: matches) {
        cv::Point2d p1 = keypointsPrevFrame[m.queryIdx].pt;
        cv::Point2d p2 = keypointsCurrentFrame[m.trainIdx].pt;

        p2dPrevFrame.push_back(p1);
        p2dCurrentFrame.push_back(p2);
    }
}

void VisualOdometry::estimatePose3D2D(std::vector<cv::Point3f>& p3d,
                                      std::vector<cv::Point2f>& p2d_PrevFrame,
                                      std::vector<cv::Point2f>& p2d_CurrentFrame,
                                      std::vector<int>& indices,
                                      Eigen::Matrix3d K, Sophus::SE3d& pose){
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs = cv::Mat::zeros(4,1,CV_64F);
    cv::Mat rvec,tvec,rot_matrix;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t(0,0,0);
    cv::eigen2cv(K, cameraMatrix);

    std::vector<int> inliers;
    bool result=cv::solvePnPRansac(p3d,p2d_PrevFrame,cameraMatrix, distCoeffs,rvec,tvec, false, 100, 4.0, 0.99, inliers);

    if (result){
        cv::Rodrigues(rvec, rot_matrix);
        cv::cv2eigen(rot_matrix, R);
        cv::cv2eigen(tvec,t);
    }

    std::sort(inliers.begin(), inliers.end());

    std::vector<cv::Point2f> p2d_PrevFrame_filtered, p2d_CurrentFrame_filtered;
    std::vector<cv::Point3f> p3d_filtered;
    std::vector<int> indices_filtered;

    for (int i = 0; i < inliers.size(); i++){
        std::vector<cv::Point3f>::iterator it_3D = p3d.begin() + inliers[i];
        std::vector<cv::Point2f>::iterator it_2D_Prev = p2d_PrevFrame.begin() + inliers[i];
        std::vector<cv::Point2f>::iterator it_2D_Curr = p2d_CurrentFrame.begin() + inliers[i];
        if (!indices.empty()){
            std::vector<int>::iterator it_2D_indices = indices.begin() + inliers[i];
            indices_filtered.push_back(*it_2D_indices);
        }

        p3d_filtered.push_back(*it_3D);
        p2d_PrevFrame_filtered.push_back(*it_2D_Prev);
        p2d_CurrentFrame_filtered.push_back(*it_2D_Curr);

    }

    std::swap(p3d, p3d_filtered);
    std::swap(p2d_CurrentFrame, p2d_CurrentFrame_filtered);
    std::swap(indices, indices_filtered);
    std::swap(p2d_PrevFrame, p2d_PrevFrame_filtered);

    Sophus::SE3d newPose = Sophus::SE3d(R, t);

    int MAX_POSE_NORM = 100;
    if (newPose.log().norm() <= MAX_POSE_NORM){
        pose = newPose;
    }
}

void VisualOdometry::estimatePose2D2D(std::vector<cv::Point2f> p2d_1, std::vector<cv::Point2f> p2d_2, Eigen::Matrix3d K, Sophus::SE3d& pose){
    cv::Mat tvec,rot_matrix;
    cv::Mat cameraMatrix;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;

    cv::eigen2cv(K, cameraMatrix);

    float focal = (K(0,0) + K(1,1)) / 2;
    cv::Point2f princip_point = cv::Point2f(K(0,2), K(1,2));

    cv::Mat E = cv::findEssentialMat(p2d_1, p2d_2, focal, princip_point);
    cv::recoverPose(E, p2d_1, p2d_2, cameraMatrix, rot_matrix, tvec);

    cv::cv2eigen(rot_matrix, R);
    cv::cv2eigen(tvec,t);

    pose = Sophus::SE3d(R, t);
}

std::vector<uchar> VisualOdometry::trackFeatures(const cv::Mat prevFrame, const cv::Mat currFrame, std::vector<cv::Point2f>& prevFramePoints, std::vector<cv::Point2f>& currFramePoints,
                                   const int thresholdNumberFeatures, bool& initialize){
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.03);
    cv::Size winSize(31,31);

    std::vector<uchar> status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(prevFrame, currFrame, prevFramePoints, currFramePoints, status, err, winSize, 3, termcrit, 0, 0.001);

    int numTracked = 0;
    for (int i = 0; i < status.size(); i++){
        if (status[i]){
            numTracked++;
        }
    }

    std::cout << "Tracked size: " << numTracked << " / " << status.size() << std::endl;
    //std::cout << prevFramePoints.size() << " " << currFramePoints.size() << std::endl;

    if (numTracked < thresholdNumberFeatures){
        initialize = true;
    } else {
        initialize = false;
    }

    return status;
}
