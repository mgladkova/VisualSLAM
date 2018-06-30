#include "VisualSLAM.h"
#include "GlobalParam.h"
#include "VisualizationToolkit.h"
#include <pangolin/pangolin.h>
#include <iostream>
#include <fstream>
#ifdef WITH_VIZ
#include <opencv2/viz.hpp>
#endif

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
#ifdef VIS_POINTCLOUD
    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(640,480,420,420,320,240,0.2,100),
                pangolin::ModelViewLookAt(-2,2,-2, 0,0,0, 0, 0, -1)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));
#endif
    float sz = 0.7;
    int width = 640, height = 480;
    Eigen::Matrix3d K = slam.getCameraMatrix();
    float fx = K(0,0);
    float fy = K(1,1);
    float cx = K(0,2);
    float cy = K(1,2);

    int k = 1;

    for (int i = 0; i < num_images; i++){

        if (i == std::pow(10, k)){
            image_name_template = image_name_template.substr(0, image_name_template.length() - 1);
            k++;
        }

        std::string image_left_name = input_left_images_path + image_name_template + std::to_string(i) + ".png";
        std::string image_right_name = input_right_images_path + image_name_template + std::to_string(i) + ".png";
        cv::Mat image_left = cv::imread(image_left_name, 0), image_left_BGR;
        cv::Mat image_right = cv::imread(image_right_name, 0);

        cv::cvtColor(image_left, image_left_BGR, CV_GRAY2BGR);

        if (image_left.cols == 0 || image_left.rows == 0){
            throw std::runtime_error("Cannot read the image with the path: " + image_left_name);
        }

        if (image_right.cols == 0 || image_right.rows == 0){
            throw std::runtime_error("Cannot read the image with the path: " + image_right_name);
        }
        Sophus::SE3d pose = slam.performFrontEndStepWithTracking(image_left, image_right, pointsCurrentFrame, pointsPrevFrame, prevImageLeft);
        plot2DPoints(image_left, pointsCurrentFrame);

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
                plotTrajectoryNextStep(window, i, translGTAccumulated, translEstimAccumulated, groundTruthPoses[i], groundTruthPoses[i-1], cumR, pose);
            }
        }
#endif

#ifdef VIS_POINTCLOUD
        std::vector<cv::Point3f> pointCloud = slam.getStructure3D();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        glPushMatrix();
        Sophus::Matrix4f m = cumPose.inverse().matrix().cast<float>();
        glMultMatrixf((GLfloat *) m.data());
        glColor3f(1, 0, 0);
        glLineWidth(2);
        glBegin(GL_LINES);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
        glEnd();
        glPopMatrix();

        glPointSize(2);
        glBegin(GL_POINTS);

        for (auto &p: pointCloud) {
            glColor3f(0.0, 0.0, 1.0);
            glVertex3d(p.x, p.y, p.z);
        }

        glEnd();
        pangolin::FinishFrame();
        usleep(5000);
#endif
    }
#ifdef VIS_POSES
    visualizeAllPoses(slam.getPoses(), slam.getCameraMatrix());
#endif
    cv::imwrite("result_trajectories.png", window);
    return 0;
}
