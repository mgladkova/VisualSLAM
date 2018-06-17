// * Class BundleAdjuster is responsible for 3D structure and camera pose optimizations


#include <Eigen/Core>
#include <Eigen/Dense>
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

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

struct ReprojectionError3D
{
        ReprojectionError3D(double observed_u, double observed_v)
                :observed_u(observed_u), observed_v(observed_v)
                {}

        template <typename T>
        bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
        {
                T p[3];
                /***** code from hk technology */
                ceres::QuaternionRotatePoint(camera_R, point, p);
                p[0] += camera_T[0];
                p[1] += camera_T[1];
                p[2] += camera_T[2];

                double fx=718.856;
                double fy=718.856;
                double cx=607.193;
                double cy=185.216;
                T xp = p[0]*fx*1./ p[2] +cx;
                T yp = p[1]*fy*1./ p[2] +cy;

                residuals[0] = xp - T(observed_u);
                residuals[1] = yp - T(observed_v);
                return true;
        }

        static ceres::CostFunction* Create(const double observed_x,
                                           const double observed_y)
        {
          return (new ceres::AutoDiffCostFunction<
                  ReprojectionError3D, 2, 4, 3, 3>(
                        new ReprojectionError3D(observed_x,observed_y)));
        }

        double observed_u;
        double observed_v;
};

class BundleAdjuster {
public:
        BundleAdjuster();
        Sophus::SE3d optimizeLocalPoseBA_ceres(std::vector<cv::Point3d> p3d,std::vector<cv::Point2d> p2d,Eigen::Matrix3d K,Sophus::SE3d pose,int iteration_times);
};
