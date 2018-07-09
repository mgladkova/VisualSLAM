#pragma once

#include "GlobalParam.h"
#include "VisualSLAM.h"
#include <mutex>
#include <thread>
#include <chrono>


class Viewer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Viewer(VisualSLAM& slam);
    void run();
    void stop();
    void finish();

    bool isStopped();
    bool isFinished();

    void drawPose();
    void draw3DStructure();
private:
    VisualSLAM* mSlam;
    std::mutex mMutexFinish;
    bool mStopped;
    bool mFinished;
    std::mutex mMutexStop;
};
