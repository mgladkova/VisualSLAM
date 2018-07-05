// * Class BundleAdjuster is responsible for optimizations of 3D structure and camera pose

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <sophus/se3.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/stereo.hpp>
#include <opencv2/core/eigen.hpp>

#include <string>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <chrono>

#include "Map.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

struct ReprojectionError{
public:
    double observ_x;
    double observ_y;

    ReprojectionError(double observ_x_new, double observ_y_new):observ_x(observ_x_new), observ_y(observ_y_new){}

    template<typename T>

    // created according to the camera model description in http://grail.cs.washington.edu/projects/bal/index.html
    bool operator()(const T* const cam, const T* const pt, T* residuals) const {
        // optical center of camera is at (0,0)
        T p[3];
        // cam[0:4] represent rotation
        T quaternion[4] = {cam[0], cam[1], cam[2], cam[3]};

        ceres::QuaternionRotatePoint(quaternion, pt, p);

        // cam[3-5] represent translation
        p[0] += cam[4];
        p[1] += cam[5];
        p[2] += cam[6];

        const T& focal = T(718.856);
        const T& cx = T(607.1928);
        const T& cy = T(185.2157);

        // normalization into
        T x = p[0] / p[2];
        T y = p[1] / p[2];

        T project_x = focal*x + cx;
        T project_y = focal*y + cy;

        residuals[0] = project_x - observ_x;
        residuals[1] = project_y - observ_y;

        return true;

    }

    static ceres::CostFunction* Create(double observ_x_new, double observ_y_new){
        ceres::CostFunction* cost_fun = new ceres::AutoDiffCostFunction<ReprojectionError, 2, 7, 3>
        (new ReprojectionError(observ_x_new, observ_y_new));
        return cost_fun;
    }
};

class BundleAdjuster {
public:
        BundleAdjuster();
        std::vector<Sophus::SE3> optimizeCameraPosesForKeyframes(Map map, int keyFrameStep, int numKeyFrames);
        void prepareDataForBA(Map map, int startFrame, int currentCameraIndex, int keyFrameStep, std::set<int> pointIndices, double* points3D, double* cameraPose);
        int getStartFrame();
        int getcurrentCameraIndex();
};
