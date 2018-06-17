#include "BundleAdjuster.h"

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix66d;

BundleAdjuster::BundleAdjuster() {}

Sophus::SE3d BundleAdjuster::optimizeLocalPoseBA_ceres(std::vector<cv::Point3d> p3d,std::vector<cv::Point2d> p2d, Eigen::Matrix3d K,Sophus::SE3d pose,int numberIterations)
{
        assert(p3d.size() == p2d.size());

        Sophus::SE3d T_esti(pose); // estimated pose
        Eigen::Matrix3d R = T_esti.so3().matrix();
        Eigen::Vector3d t = T_esti.translation();

        Eigen::Quaterniond q_rotation(R) ;

        //local BA
        ceres::Problem problem;
        ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
        //double array for ceres
        int numPoints3D = p3d.size();
        double translation[3] = {t[0], t[1], t[2]};
        double points3D[numPoints3D][3];

        double camera[8];
        camera[0] = q_rotation.w();
        camera[1] = q_rotation.x();
        camera[2] = q_rotation.y();
        camera[3] = q_rotation.z();

        camera[4] = K(0,0);
        camera[5] = K(1,1);
        camera[6] = K(0,2);
        camera[7] = K(1,2);

        // not required, keeping it for reference
        // double cameraRotation[4] = {q_rotation.w(), q_rotation.x(), q_rotation.y(), q_rotation.z()};
        // problem.AddParameterBlock(cameraRotation, 4, local_parameterization);
        // problem.AddParameterBlock(translation, 3);

        // add 3 D points
        for (int i = 0; i < numPoints3D; ++i){
            points3D[i][0] = p3d[i].x;
            points3D[i][1] = p3d[i].y;
            points3D[i][2] = p3d[i].z;
        }

        for (int j = 0; j < p2d.size(); j++){
            ceres::CostFunction* cost_function = ReprojectionError3D::Create(p2d[j].x, p2d[j].y);
            problem.AddResidualBlock(cost_function, NULL, camera, translation, points3D[j]);
        }

        // Solver::Options options;
        ceres::Solver::Options options;
        options.preconditioner_type = ceres::SCHUR_JACOBI;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = false;
        // options.max_solver_time_in_seconds = 0.2;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        //std::cout << summary.BriefReport() << "\n";
        //std::cout << "Final rotation: " << rotation << " Final translation: " << translation << "\n";
        //std::cout << "Final   m: " << m << " c: " << c << "\n";

        Eigen::Quaterniond quaterniond_(camera[0], camera[1], camera[2], camera[3]);
        Eigen::Vector3d translation_(translation[0],translation[1],translation[2]);
        Sophus::SE3d cameraTransform(quaterniond_,translation_);
        //std::cout << SE3_quaternion.matrix() << std::endl;
        return cameraTransform;
}
