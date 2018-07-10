#include "Viewer.h"

Viewer::Viewer(VisualSLAM& slam){
    mSlam = &slam;
}

void Viewer::run(){
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
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        Visualization3D_display.Activate(Visualization3D_camera);
        glClearColor(0.0f,0.0f,0.0f,0.0f);

        int cameraIndex;
        Sophus::SE3d camera, gtCamera;
        std::vector<cv::Point3f> structure3d;
        std::vector<int> obsIndices;

        mSlam->getDataFromImageLeftForDrawing(cameraIndex, camera, structure3d, obsIndices, gtCamera);

        drawPose(camera);
        drawPose(gtCamera);

        draw3DStructure(structure3d);

        drawConnections(camera, obsIndices, structure3d);

        pangolin::FinishFrame();

        usleep(7000);
    }
}

void Viewer::drawConnections(Sophus::SE3d cameraPose, std::vector<int> obsIndices, std::vector<cv::Point3f> structure3d){
    glLineWidth(1);
    glBegin(GL_LINES);

    glColor3f(0,1,0);
    glBegin(GL_LINES);
    for(auto it = obsIndices.begin(); it != obsIndices.end(); it++){
        int pointIndex = *it;
        if (pointIndex < 0 || pointIndex >= structure3d.size()){
            std::cerr << "drawConnections() : OUT OF BOUNDS , PointIndex: " << pointIndex << std::endl;
            //throw std::runtime_error("drawConnections() : 3D point index is out of bounds");
            continue;
        }

        //std::cout << "POINT INDEX " << pointIndex << " seen by camera at " << structure3d[pointIndex] << std::endl;

        glVertex3f((GLfloat) cameraPose.inverse().translation()[0],(GLfloat) cameraPose.inverse().translation()[1], (GLfloat) cameraPose.inverse().translation()[2]);
        glVertex3f((GLfloat) structure3d[pointIndex].x,(GLfloat) structure3d[pointIndex].y, (GLfloat) structure3d[pointIndex].z);
    }
    glEnd();
}

void Viewer::draw3DStructure(std::vector<cv::Point3f> pointCloud){
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

void Viewer::drawPose(Sophus::SE3d cameraPose){
    const float w = 0.2;
    const float h = w*0.75;
    const float z = w*0.6;

    Sophus::Matrix4f m = cameraPose.inverse().matrix().cast<float>();

    glPushMatrix();
    glMultMatrixf((GLfloat *) m.data());
    glColor3f(1, 0, 0);
    glLineWidth(2);
    glBegin(GL_LINES);

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
