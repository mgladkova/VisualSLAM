#pragma once
#include "VisualSLAM.h"

class Viewer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Viewer(VisualSLAM& slam);
    void run();

    void drawPose(Sophus::SE3d cameraPose);
    void draw3DStructure(std::vector<cv::Point3f> structure3D);

    void drawConnections(Sophus::SE3d cameraPose, std::vector<int> obsIndices, std::vector<cv::Point3f> structure3D);

private:
    VisualSLAM* mSlam;
};
