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

//Sophus::SE3 BundleAdjuster::optimizeLocalPoseBA(std::vector<cv::Point3d> p3d,std::vector<cv::Point2d> p2d, Eigen::Matrix3d K,Sophus::SE3 pose,int numberIterations){
//    assert(p3d.size() == p2d.size());
//    double cost = 0, lastCost = 0;
//    int nPoints = p3d.size();

//    double fx = K(0,0);
//    double fy = K(1,1);

//    Sophus::SE3 T_esti(pose); // estimated pose

//    for (int iter = 0; iter < numberIterations; iter++) {
//        Matrix66d H = Matrix66d::Zero();
//        Vector6d b = Vector6d::Zero();
//        cost = 0;
//        // compute cost
//        for (int i = 0; i < nPoints; i++) {
//            // compute cost for p3d[I] and p2d[I]
//            Eigen::Vector3d point3DEigen(p3d[i].x, p3d[i].y, p3d[i].z);

//            Eigen::Vector2d point2DEigen(p2d[i].x, p2d[i].y);

//            Eigen::Vector3d pointTransformed = T_esti*point3DEigen;

//            Eigen::Vector3d pointReprojected = K * pointTransformed;
//            pointReprojected /= pointTransformed[2];

//            Eigen::Vector2d e(0,0);    // error
//            e[0]=point2DEigen[0] - pointReprojected[0];
//            e[1]=point2DEigen[1] - pointReprojected[1];

//            // compute jacobian
//            Eigen::Matrix<double, 2, 6> J;

//            double z = pointTransformed[2];
//            double y = pointTransformed[1];
//            double x = pointTransformed[0];

//            J(0,0) = fx/z;
//            J(0,1) = 0.0;
//            J(0,2) = -(fx*x)/(z*z);
//            J(0,3) = -(fx*x*y)/(z*z);
//            J(0,4) = fx + ((fx*x*x)/(z*z));
//            J(0,5) = -(fx*y)/z;
//            J(1,0) = 0.0;
//            J(1,1) = fy/z;
//            J(1,2) = -(fy*y)/(z*z);
//            J(1,3) = -fy - ((fy*y*y)/(z*z));
//            J(1,4) = (fy*x*y)/(z*z);
//            J(1,5) = (fy*x)/z;

//            J=-J;

//            H += J.transpose() * J;
//            b += -J.transpose() * e;

//            cost += 0.5 *(e[0]*e[0] + e[1]*e[1]);
//            // cout<<"cost on line is : "<<cost<<endl;
//        }

//        // solve dx
//        Vector6d dx = H.inverse()*b;    //the caculation of H*dx=b is right

//        if (std::isnan(dx[0])) {
//            std::cout << "result is nan!" << std::endl;
//            break;
//        }

//        if (iter > 0 && cost >= lastCost) {
//            // cost increase, update is not good
//            std::cout << "cost: " << cost << ", last cost: " << lastCost << std::endl;
//            break;
//        }

//        // update your estimation
//        T_esti = Sophus::SE3::exp(dx)*T_esti;

//        lastCost = cost;

//        std::cout << "iteration " << iter << " cost=" << std::cout.precision(12) << cost << std::endl;
//        //std::cout << "estimated pose: \n" << T_esti.matrix() << std::endl;

//    }
//    return T_esti;
//}
