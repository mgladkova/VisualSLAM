#include "VisualSLAM.h"

VisualSLAM::VisualSLAM() {
    K = Eigen::Matrix3d::Identity();
}

Sophus::SE3d VisualSLAM::getPose(int index) {
    if (index < 0 || index >= historyPoses.size()){
        throw std::runtime_error("VisualSLAM::getPose() : Index out of bounds");
    }

    return historyPoses.at(index);
}

int VisualSLAM::getNumberPoses() const{
    return historyPoses.size();
}

Eigen::Matrix3d VisualSLAM::getCameraMatrix() const {
    return K;
}

double VisualSLAM::getFocalLength() const {
    return (K(0,0) + K(1,1)) / 2.0;
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

void VisualSLAM::performFrontEndStep(cv::Mat image_left, cv::Mat image_right){
    std::vector<cv::KeyPoint> keypoints_new;
	cv::Mat descriptors_new;

    VO.extractORBFeatures(image_left, keypoints_new, descriptors_new);
    KeyFrame refFrame = VO.getReferenceFrame();
    cv::Mat disparity_map = VO.getDisparityMap(image_left, image_right);

    if (refFrame.keypoints.empty()){
        VO.setReferenceFrame(image_left, disparity_map, keypoints_new, descriptors_new);
		return;
	}

	std::vector<cv::DMatch> matches = VO.findGoodORBFeatureMatches(keypoints_new, descriptors_new);

	// Draw top matches

    cv::Mat imMatches;
    cv::drawMatches(refFrame.image, refFrame.keypoints, image_left, keypoints_new, matches, imMatches);
    cv::imshow("Matches", imMatches);
    cv::waitKey(0);

    std::vector<cv::Point3d> p3d_prevFrame;
    std::vector<cv::Point2d> p2d_currFrame;
    VO.get3D2DCorrespondences(keypoints_new, matches, p3d_prevFrame, p2d_currFrame, K);

    VO.estimatePose3D2D(p3d_prevFrame, p2d_currFrame, K);
    VO.setReferenceFrame(image_left, disparity_map, keypoints_new, descriptors_new);


    // each motion bundle adjustment (reference image's 3D point and matched image's 2D point)
    Sophus::SE3d new_pose = BA.optimizeLocalPoseBA(p3d_prevFrame,p2d_currFrame, K, VO.getPose(), 50);
    historyPoses.push_back(new_pose);
    std::cout << "After optimizations:\n" << new_pose.matrix() << std::endl;
}

void VisualSLAM::runBackEndRoutine(){
	// TODO
}

void VisualSLAM::update(){
	//TODO
}

void VisualSLAM::visualizeAllPoses(){
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("VisualSLAM Viewer", 1024, 768);
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

    double fx = K(0,0);
    double fy = K(1,1);
    double cx = K(0,2);
    double cy = K(1,2);

    while (pangolin::ShouldQuit() == false)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        // draw poses
        float sz = 0.1;
        int width = 640, height = 480;
        for (auto &Tcw: historyPoses)
        {
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
