#include "Viewer.h"

Viewer::Viewer(VisualSLAM& slam){
    mSlam = &slam;
    mStopped = true;
    mFinished = true;

}

void Viewer::run(){
    mFinished = false;
    mStopped = false;

    int w = 640, h = 480;

    pangolin::CreateWindowAndBind("VisualSLAM Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState Visualization3D_camera(
            pangolin::ProjectionMatrix(w,h,400,400,w/2,h/2,0.1,1000),
            pangolin::ModelViewLookAt(-0,-5,-10, 0,0,0, pangolin::AxisNegY)
            );

    pangolin::View& Visualization3D_display = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -w/(float)h)
            .SetHandler(new pangolin::Handler3D(Visualization3D_camera));

    while(pangolin::ShouldQuit() == false){
        int index = mSlam->getNumberPoses() - 1;
        if (index < 0){
            continue;
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        Visualization3D_display.Activate(Visualization3D_camera);
        glClearColor(0.0f,0.0f,0.0f,0.0f);
        drawPose(index);

        draw3DStructure();

        drawConnections(index);

        pangolin::FinishFrame();

        if(isFinished())
            break;

        if(isStopped()){
            while(isStopped()){
                usleep(3000);
            }
        }
    }

    finish();
}

void Viewer::drawConnections(int cameraIndex){
    if (cameraIndex >= mSlam->getNumberPoses() || cameraIndex < 0){
        return;
    }

    Eigen::Vector3d cameraTransl = mSlam->getPose(cameraIndex).inverse().translation();
    std::vector<cv::Point3f> structure3d = mSlam->getStructure3D();
    std::vector<std::pair<int, cv::Point2f>> observations = mSlam->getObservationsForCamera(cameraIndex);

    glLineWidth(1);
    glBegin(GL_LINES);

    glColor3f(0,1,0);
    glBegin(GL_LINES);
    for(unsigned int i = 0; i < observations.size(); i++){
        glVertex3f((GLfloat) cameraTransl[0],(GLfloat) cameraTransl[1], (GLfloat) cameraTransl[2]);
        int pointIndex = observations[i].first;
        if (pointIndex < 0 || pointIndex >= structure3d.size()){
            throw std::runtime_error("drawConnections() : 3D point index is out of bounds");
        }

        glVertex3f((GLfloat) structure3d[pointIndex].x,(GLfloat) structure3d[pointIndex].y, (GLfloat) structure3d[pointIndex].z);
    }
    glEnd();
}

void Viewer::draw3DStructure(){
    std::vector<cv::Point3f> pointCloud = mSlam->getStructure3D();
    int pointSize = 4;

    if (pointCloud.empty()){
        return;
    }

    glPointSize(pointSize);
    glBegin(GL_POINTS);

    for (auto &p: pointCloud) {
        glColor3f(0.0, 0.0, 1.0);
        glVertex3d(p.x, p.y, p.z);
    }

    glEnd();
}

void Viewer::drawPose(int cameraIndex){
    float sz = 0.7;
    int width = 640, height = 480;
    Eigen::Matrix3d K = mSlam->getCameraMatrix();
    float fx = K(0,0);
    float fy = K(1,1);
    float cx = K(0,2);
    float cy = K(1,2);

    const float w = 0.2;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();
    Sophus::Matrix4f m = mSlam->getPose(cameraIndex).inverse().matrix().cast<float>();
    glMultMatrixf((GLfloat *) m.data());
    glColor3f(1, 0, 0);
    glLineWidth(2);
    glBegin(GL_LINES);
//    glVertex3f(0, 0, 0);
//    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
//    glVertex3f(0, 0, 0);
//    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
//    glVertex3f(0, 0, 0);
//    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
//    glVertex3f(0, 0, 0);
//    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
//    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
//    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
//    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
//    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
//    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
//    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
//    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
//    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glEnd();
    glPopMatrix();
}

void Viewer::stop(){
    std::unique_lock<std::mutex> lock(mMutexStop);
    mStopped = true;
}

void Viewer::finish(){
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mFinished = true;
}

bool Viewer::isStopped(){
    std::unique_lock<std::mutex> lock(mMutexStop);
    return mStopped;
}

bool Viewer::isFinished(){
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mFinished;
}
