#include "BundleAdjuster.h"

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix66d;

BundleAdjuster::BundleAdjuster() {}

Sophus::SE3d BundleAdjuster::optimizeLocalPoseBA(std::vector<cv::Point3d> p3d,std::vector<cv::Point2d> p2d, Eigen::Matrix3d K,Sophus::SE3d pose,int numberIterations){
    assert(p3d.size() == p2d.size());
    double cost = 0, lastCost = 0;
    int nPoints = p3d.size();

    double fx = K(0,0);
    double fy = K(1,1);

    Sophus::SE3d T_esti(pose); // estimated pose

    for (int iter = 0; iter < numberIterations; iter++) {
        Matrix66d H = Matrix66d::Zero();
        Vector6d b = Vector6d::Zero();
        cost = 0;
        // compute cost
        for (int i = 0; i < nPoints; i++) {
            // compute cost for p3d[I] and p2d[I]
            Eigen::Vector3d point3DEigen(p3d[i].x, p3d[i].y, p3d[i].z);

            Eigen::Vector2d point2DEigen(p2d[i].x, p2d[i].y);

            Eigen::Vector3d pointTransformed = T_esti*point3DEigen;

            Eigen::Vector3d pointReprojected = K * pointTransformed;
            pointReprojected /= pointTransformed[2];

            Eigen::Vector2d e(0,0);    // error
            e[0]=point2DEigen[0] - pointReprojected[0];
            e[1]=point2DEigen[1] - pointReprojected[1];

            // compute jacobian
            Eigen::Matrix<double, 2, 6> J;

            double z = pointTransformed[2];
            double y = pointTransformed[1];
            double x = pointTransformed[0];

            J(0,0) = fx/z;
            J(0,1) = 0.0;
            J(0,2) = -(fx*x)/(z*z);
            J(0,3) = -(fx*x*y)/(z*z);
            J(0,4) = fx + ((fx*x*x)/(z*z));
            J(0,5) = -(fx*y)/z;
            J(1,0) = 0.0;
            J(1,1) = fy/z;
            J(1,2) = -(fy*y)/(z*z);
            J(1,3) = -fy - ((fy*y*y)/(z*z));
            J(1,4) = (fy*x*y)/(z*z);
            J(1,5) = (fy*x)/z;

            J=-J;

            H += J.transpose() * J;
            b += -J.transpose() * e;

            cost += 0.5 * e.norm();
            // cout<<"cost on line is : "<<cost<<endl;
        }

        // solve dx
        Vector6d dx = H.inverse()*b;    //the caculation of H*dx=b is right

        if (std::isnan(dx[0])) {
            std::cout << "result is nan!" << std::endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            // cost increase, update is not good
            std::cout << "cost: " << cost << ", last cost: " << lastCost << std::endl;
            break;
        }

        // update your estimation
        T_esti = Sophus::SE3d::exp(dx)*T_esti;

        lastCost = cost;

        std::cout << "iteration " << iter << " cost=" << std::cout.precision(12) << cost << std::endl;
        //std::cout << "estimated pose: \n" << T_esti.matrix() << std::endl;

    }
    return T_esti;
}
