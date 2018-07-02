#include "VisualizationToolkit.h"

void plot2DPoints(cv::Mat image, std::vector<cv::Point2f> points2d){
    if (points2d.empty()){
        std::cout << "plot2DPoints(): No features to plot!" << std::endl;
        return;
    }

    cv::Mat imageCopy;
    if (image.cols == 0){
        imageCopy = cv::Mat(500, 500, CV_8UC3);
    } else{
        cv::cvtColor(image, imageCopy, CV_GRAY2BGR);
    }

    for (auto& pt : points2d){
        cv::circle(imageCopy, pt, 3, cv::Scalar(0,0,255), -1);
    }
    cv::imshow("Detected features", imageCopy);
}

void plot2DPoints(cv::Mat image, std::vector<cv::KeyPoint> keypoints){
    std::vector<cv::Point2f> points2d;

    for (auto& kPt : keypoints){
        points2d.push_back(kPt.pt);
    }

    plot2DPoints(image, points2d);
}

void plotTrajectoryNextStep(cv::Mat& window, int index, Eigen::Vector3d& translGTAccumulated, Eigen::Vector3d& translEstimAccumulated,
                            Sophus::SE3d groundTruthPose, Sophus::SE3d groundTruthPrevPose, Eigen::Matrix3d& cumR, Sophus::SE3d estimPose,  Sophus::SE3d estimPrevPose){
    int offsetX = 300;
    int offsetY = 500;

    Sophus::SE3d pose = estimPose.inverse();
    Sophus::SE3d prevPose = estimPrevPose.inverse();
    
    if (index == 0){
        translGTAccumulated = groundTruthPose.translation();
        translEstimAccumulated = pose.translation();
    } else {
        translGTAccumulated = translGTAccumulated + (groundTruthPose.so3().inverse()*groundTruthPrevPose.so3())*(groundTruthPose.translation() - groundTruthPrevPose.translation());
        translEstimAccumulated = translEstimAccumulated + (pose.so3().inverse()*prevPose.so3())*(pose.translation() - prevPose.translation());
    }
    cv::circle(window, cv::Point2d(offsetX + translGTAccumulated[0], offsetY - translGTAccumulated[2]), 3, cv::Scalar(0,0,255), -1);
    cv::circle(window, cv::Point2f(offsetX + translEstimAccumulated[0], offsetY - translEstimAccumulated[2]), 3, cv::Scalar(0,255,0), -1);
    cv::imshow("Trajectory", window);
    cv::waitKey(3);
    cumR = cumR*pose.so3().matrix();
}
