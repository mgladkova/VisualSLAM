#include "VisualSLAM.h"
#pragma once

class Viewer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Viewer(VisualSLAM& slam);
    void run(int numFrames);
    void stop();
    void finish();
    void resume();

    bool isStopped();
    bool isFinished();

    void drawPose(int cameraIndex);
    void draw3DStructure();

    void drawConnections(int cameraIndex);

private:
    VisualSLAM* mSlam;
    std::mutex mMutexFinish;
    bool mStopped;
    bool mFinished;
    std::mutex mMutexStop;
};
