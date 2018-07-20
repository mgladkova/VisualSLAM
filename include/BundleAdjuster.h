// * Class BundleAdjuster is responsible for optimizations of 3D structure and camera pose

#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <sophus/se3.hpp>

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

struct Pose3d {
  Eigen::Vector3d p;
  Eigen::Quaterniond q;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct ReprojectionError{
private:
    double observ_x;
    double observ_y;

    // The measurement for the position of B relative to A in the A frame.
    Pose3d t_ab_measured;
    // The square root of the measurement information matrix.
    Eigen::Matrix<double, 7, 7> sqrt_information;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionError(double observ_x_new,
                      double observ_y_new,
                      Pose3d t_ab_measured,
                      Eigen::Matrix<double, 7, 7> sqrt_information):observ_x(observ_x_new),
                                                                    observ_y(observ_y_new),
                                                                    t_ab_measured(t_ab_measured),
                                                                    sqrt_information(sqrt_information){}

    template<typename T>

    // created according to the camera model description in http://grail.cs.washington.edu/projects/bal/index.html
    bool operator()(const T* const cam_left, const T* cam_right, const T* const pt, T* residuals) const {
        // optical center of camera is at (0,0)
        T p[3];
        // cam[0:4] represent rotation
        T quaternion[4] = {cam_left[0], cam_left[1], cam_left[2], cam_left[3]};

        ceres::QuaternionRotatePoint(quaternion, pt, p);

        // cam[3-5] represent translation
        p[0] += cam_left[4];
        p[1] += cam_left[5];
        p[2] += cam_left[6];

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

        Eigen::Matrix<T, 3, 1> p_a(cam_left[4], cam_left[5], cam_left[6]);
        Eigen::Quaternion<T> q_a(cam_left[0], cam_left[1], cam_left[2], cam_left[3]);

        Eigen::Matrix<T, 3, 1> p_b(cam_right[4], cam_right[5], cam_right[6]);
        Eigen::Quaternion<T> q_b(cam_right[0], cam_right[1], cam_right[2], cam_right[3]);

        // Compute the relative transformation between the two frames.
        Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();
        Eigen::Quaternion<T> q_ab_estimated = q_a_inverse * q_b;

        // Represent the displacement between the two frames in the A frame.
        Eigen::Matrix<T, 3, 1> p_ab_estimated = q_a_inverse * (p_b - p_a);

        // Compute the error between the two orientation estimates.
        Eigen::Quaternion<T> delta_q =
            t_ab_measured.q.template cast<T>() * q_ab_estimated.conjugate();

        // Compute the residuals.
        // [ position         ]   [ delta_p          ]
        // [ orientation (3x1)] = [ 2 * delta_q(0:2) ]
        residuals[2] = sqrt_information(0,0)*(p_ab_estimated[0] - t_ab_measured.p.template cast<T>()[0]);
        residuals[3] = sqrt_information(1,1)*(p_ab_estimated[1] - t_ab_measured.p.template cast<T>()[1]);
        residuals[4] = sqrt_information(2,2)*(p_ab_estimated[2] - t_ab_measured.p.template cast<T>()[2]);
        residuals[5] = sqrt_information(3,3)*(T(2.0) * delta_q.w());
        residuals[6] = sqrt_information(4,4)*(T(2.0) * delta_q.x());
        residuals[7] = sqrt_information(5,5)*(T(2.0) * delta_q.y());
        residuals[8] = sqrt_information(6,6)*(T(2.0) * delta_q.z());

        return true;

    }

    static ceres::CostFunction* Create(double observ_x_new, double observ_y_new,
                                       Pose3d t_ab_measured,
                                       Eigen::Matrix<double, 7, 7> sqrt_information){
        ceres::CostFunction* cost_fun = new ceres::AutoDiffCostFunction<ReprojectionError, 9, 7, 7, 3>
        (new ReprojectionError(observ_x_new, observ_y_new, t_ab_measured, sqrt_information));
        return cost_fun;
    }
};

class BundleAdjuster {
public:
        BundleAdjuster();
        bool performBAWithKeyFrames(Map& map_left, Map& map_right, int keyFrameStep, int numKeyFrames);
        //bool performPoseGraphOptimization(Map& map_left, Map& map_right, int keyFrameStep, int numKeyFrames);
        void prepareDataForBA(Map& map, int startFrame, int currentCameraIndex, int keyFrameStep, std::set<int> pointIndices, double* points3D, double* cameraPose);
};
