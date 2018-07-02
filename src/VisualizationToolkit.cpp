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

void computeAndShowPointCloud(const cv::Mat image_left, const cv::Mat disparity, const float baseline, Eigen::Matrix3d K) {
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

void plotTrajectoryNextStep(cv::Mat& window, int index, Eigen::Vector3d& translGTAccumulated, Eigen::Vector3d& translEstimAccumulated,
                            Sophus::SE3d groundTruthPose, Sophus::SE3d groundTruthPrevPose, Eigen::Matrix3d& cumR, Sophus::SE3d estimPose,  Sophus::SE3d estimPrevPose){
    int offsetX = 200;
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
    cv::circle(window, cv::Point2d(offsetX + translGTAccumulated[2], offsetX + translGTAccumulated[1]), 3, cv::Scalar(0,0,255), -1);
    cv::circle(window, cv::Point2f(offsetX + translEstimAccumulated[2], offsetY + translEstimAccumulated[1]), 3, cv::Scalar(0,255,0), -1);
    //cv::circle(window, cv::Point2d(offsetX + translGTAccumulated[0], offsetY - translGTAccumulated[2]), 3, cv::Scalar(0,0,255), -1);
    //cv::circle(window, cv::Point2f(offsetX + translEstimAccumulated[0], offsetY - translEstimAccumulated[2]), 3, cv::Scalar(0,255,0), -1);
    cv::imshow("Trajectory", window);
    cv::waitKey(3);
    cumR = cumR*pose.so3().matrix();
}

void visualizeAllPoses(std::vector<Sophus::SE3d> historyPoses, Eigen::Matrix3d K){
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("VisualSLAM Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
                pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, 1.0, 0.0)
                );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    double fx = K(0,0);
    double fy = K(1,1);
    double cx = K(0,2);
    double cy = K(1,2);

    while (pangolin::ShouldQuit() == false){
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        // draw poses
        float sz = 0.1;
        int width = 640, height = 480;
        for (auto &Tcw: historyPoses){
            glPushMatrix();
            Sophus::Matrix4f m = Tcw.inverse().matrix().cast<float>();
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
        }

        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}

void showPointCloud(const std::vector<cv::Point3f> points3D) {

    if (points3D.empty()) {
        throw std::runtime_error("Point cloud is empty!");
    }

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

        for (auto &p: points3D) {
            glColor3f(0.0, 0.0, 1.0);
            glVertex3d(p.x, p.y, p.z);
        }

        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}
