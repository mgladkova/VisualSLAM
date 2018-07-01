#include "VisualSLAM.h"
#include <mutex>
#include <thread>
#include <chrono>

#pragma once

class Viewer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Viewer(VisualSLAM& slam);
    void run();
    void stop();
    void finish();

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
