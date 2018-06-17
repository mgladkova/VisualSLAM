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
//        cout<<"Quaterniond: "<<endl<<q_rotation.coeffs()<<endl;

        //local BA
        ceres::Problem problem;
        ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
        //cout << " begin local BA " << endl;
        // for (int i = 0; i < frame_num; i++)
        // {
        int i=0;
            //double array for ceres
            int position_num=p3d.size();
            double rotation[1][4];
            double translation[1][3];
            double position[position_num][3];

            translation[i][0] = t[0];
            translation[i][1] = t[1];
            translation[i][2] = t[2];
            rotation[i][0] = q_rotation.w();
            rotation[i][1] = q_rotation.x();
            rotation[i][2] = q_rotation.y();
            rotation[i][3] = q_rotation.z();
            problem.AddParameterBlock(rotation[i], 4, local_parameterization);
            problem.AddParameterBlock(translation[i], 3);
            // if (i == l)
            // {
                // problem.SetParameterBlockConstant(rotation[i]);
            // }
            // if (i == l || i == frame_num - 1)
            // {
                // problem.SetParameterBlockConstant(translation[i]);
            // }
        // }
            // add 3 D points
            for (int i = 0; i < position_num; ++i)
            {
                position[i][0]=p3d[i].x;
                position[i][1]=p3d[i].y;
                position[i][2]=p3d[i].z;
            }

            for (int j = 0; j < p2d.size(); j++)
            {
                int l = 0;
                ceres::CostFunction* cost_function = ReprojectionError3D::Create(
                                                    p2d[j].x,
                                                    p2d[j].y);

                problem.AddResidualBlock(cost_function, NULL, rotation[l], translation[l],
                                        position[j]);
            }

        // }
        // Solver::Options options;
        Eigen::Quaterniond quaterniond_0(rotation[0][0],rotation[0][1],rotation[0][2],rotation[0][3]);
        Eigen::Vector3d translation_0(translation[0][0],translation[0][1],translation[0][2]);
        Sophus::SE3d SE3_quaternion0(quaterniond_0,translation_0);
//        cout<<SE3_quaternion0.matrix()<<endl;

        ceres::Solver::Options options;
        options.preconditioner_type = ceres::SCHUR_JACOBI;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        // options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = false;
        // options.max_solver_time_in_seconds = 0.2;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
//        std::cout << summary.BriefReport() << "\n";
      // std::cout << "Final rotation: " << rotation << " Final translation: " << translation << "\n";
      // std::cout << "Final   m: " << m << " c: " << c << "\n";
        Eigen::Quaterniond quaterniond_(rotation[0][0],rotation[0][1],rotation[0][2],rotation[0][3]);
        Eigen::Vector3d translation_(translation[0][0],translation[0][1],translation[0][2]);
        Sophus::SE3d SE3_quaternion(quaterniond_,translation_);
//        cout<<SE3_quaternion.matrix()<<endl;
        return SE3_quaternion;
}
