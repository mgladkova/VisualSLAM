#include "VisualSLAM.h"
#include "Viewer.h"
#include "GlobalParam.h"

#include <iostream>
#include <fstream>


int main(int argc, char** argv){

    if (argc < 5){
        std::cout << "Usage: ./slam <input_left_image_directory> <input_right_image_directory> <camera_intrinsics_file_path> <num_images> [ground_truth_data_path]" << std::endl;
        exit(1);
    }

    std::string input_left_images_path = argv[1];
    std::string input_right_images_path = argv[2];
    std::string camera_intrinsics_path = argv[3];
    int num_images = std::stoi(argv[4]);
    std::string image_name_template = "00000";

    if (num_images <= 0)
    {
        throw std::runtime_error("The number of image pairs is invalid");
    }

    VisualSLAM slam;
    slam.readCameraIntrisics(camera_intrinsics_path);
    Eigen::Vector3d translGTAccumulated, translEstimAccumulated(1,1,1);
    std::vector<Sophus::SE3d> groundTruthPoses;

    if (argc >= 6){
        std::string ground_truth_path = argv[5];
        slam.readGroundTruthData(ground_truth_path, num_images, groundTruthPoses);
    }

    cv::Mat window = cv::Mat::zeros(1000, 1000, CV_8UC3);
    cv::Mat prevImageLeft;
    std::vector<cv::Point2f> pointsCurrentFrame, pointsPrevFrame;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    Eigen::Matrix3d cumR = Eigen::Matrix3d::Identity();

    int k = 1;

    std::vector<cv::Mat> images_left(num_images), images_right(num_images);

    for (int i = 0; i < num_images; i++){
        if (i == std::pow(10, k)){
            image_name_template = image_name_template.substr(0, image_name_template.length() - 1);
            k++;
        }

        std::string image_left_name = input_left_images_path + image_name_template + std::to_string(i) + ".png";
        std::string image_right_name = input_right_images_path + image_name_template + std::to_string(i) + ".png";
        cv::Mat image_left = cv::imread(image_left_name, 0);
        cv::Mat image_right = cv::imread(image_right_name, 0);

        if (image_left.cols == 0 || image_left.rows == 0){
            throw std::runtime_error("Cannot read the image with the path: " + image_left_name);
        }

        if (image_right.cols == 0 || image_right.rows == 0){
            throw std::runtime_error("Cannot read the image with the path: " + image_right_name);
        }

        image_left.copyTo(images_left[i]);
        image_right.copyTo(images_right[i]);

    }

    for (int i = 0; i < num_images; i++){
        Sophus::SE3d pose = slam.performFrontEndStepWithTracking(images_left[i], images_right[i], pointsCurrentFrame, pointsPrevFrame, prevImageLeft);
        //plot2DPoints(images_left[i], pointsCurrentFrame);

        Sophus::SE3d cumPose;
        if (i != 0){
            cumPose = slam.getPose(i);
        }
#ifdef VIS_TRAJECTORY
        if (!groundTruthPoses.empty() && i < groundTruthPoses.size()){
            if (i == 0){
                Sophus::SE3d groundTruthPrevPose = Sophus::SE3d(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0,0,0));
                plotTrajectoryNextStep(window, i, translGTAccumulated, translEstimAccumulated, groundTruthPoses[i], groundTruthPrevPose, cumR, pose);
            } else {
                std::cout << "Frame " << i << " / " << groundTruthPoses.size() << std::endl;
                Sophus::SE3d prevCumPose = slam.getPose(i-1);
                plotTrajectoryNextStep(window, i, translGTAccumulated, translEstimAccumulated, groundTruthPoses[i], groundTruthPoses[i-1], cumR, cumPose, prevCumPose);
            }
        }
#endif
    }
#ifdef VIS_POSES
    visualizeAllPoses(slam.getPoses(), slam.getCameraMatrix());
#endif

    Viewer* viewer = new Viewer(slam);
    viewer->run(num_images);
    //std::thread* viewerThread = new std::thread(&Viewer::run, viewer);

    //viewerThread->join();
    delete viewer;

    cv::imwrite("result_trajectories.png", window);
    return 0;
}
