#include "BundleAdjuster.h"


BundleAdjuster::BundleAdjuster() {}

Sophus::SE3 BundleAdjuster::Motion_BA(std::vector<cv::Point3d> p3d,std::vector<cv::Point2d> p2d,Eigen::Matrix3d K,Sophus::SE3 pose,int iteration_times){
       /*  Motion BA  TEST*/

//       Eigen::Matrix3d K;
//       K<<cam.fx, 0, cam.cx, 0, cam.fy, cam.cy, 0, 0, 1;
//       std::cout<<K<<std::endl;
      int iterations = iteration_times;
      // std::cout<<"error here"<<std::endl;
      assert(p3d.size() == p2d.size());
      std::cout<<"error here "<<p3d.size()<<" "<<p2d.size()<<std::endl;
      double cost = 0, lastCost = 0;
      int nPoints = p3d.size();
      Sophus::SE3 T_esti; // estimated pose
      T_esti=pose;
      for (int iter = 0; iter < iterations; iter++) {

       Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
       typedef Eigen::Matrix<double, 6, 1> Vector6d;
       Vector6d b = Vector6d::Zero();
//       std::cout<<"the T_esti: "<<T_esti.matrix()<<std::endl;
       cost = 0;
       // compute cost
       for (int i = 0; i < nPoints; i++) {
           // compute cost for p3d[I] and p2d[I]
           Eigen::Vector2d e(0,0);    // error
           Eigen::Vector3d p_estimate;
           Eigen::Matrix3d R ;
           Eigen::Vector3d t;
           // cout<<T_esti.matrix()<<endl;
           Eigen::Matrix4d T=T_esti.matrix();

           R << T(0,0),T(0,1),T(0,2),
                T(1,0),T(1,1),T(1,2),
                T(2,0),T(2,1),T(2,2);
           t << T(0,3),T(1,3),T(2,3);
           // std::cout<<" T is : "<<std::endl<<T<<std::endl;
           // std::cout<<"p3d: "<<(double)p3d[i].x<<std::endl;
           // cout<<"R is :: "<<endl<<R<<endl;
           // cout<<"t is::"<<endl<<t<<endl;
           Eigen::Vector3d P3D;
           Eigen::Vector2d P2D;
           P3D<<(double)p3d[i].x,(double)p3d[i].y,(double)p3d[i].z;
           P2D<<(double)p2d[i].x,(double)p2d[i].y;
           Eigen::Vector3d P_after_move =R*P3D+t;//R*p3d[i] +t;
           p_estimate=K*P_after_move*1./P_after_move[2];//p3d[i][2];
           e[0]=P2D[0]-p_estimate[0];
           e[1]=P2D[1]-p_estimate[1];
           // e[0]=p2d[i][0]-p_estimate[0];
           // e[1]=p2d[i][1]-p_estimate[1];
           // std::cout<<"e : "<<std::endl<<e<<std::endl;
           // compute jacobian
           Eigen::Matrix<double, 2, 6> J;
           double x(P_after_move(0)),y(P_after_move(1)),z(P_after_move(2));
           double zz=z*z;
           double yy=y*y;
           double xx=x*x;
           double xy=x*y;
//            double fx=cam.fx, fy=cam.fy, cx=cam.fx,cy=cam.cy;
           double fx=K(0,0);
           double fy=K(1,1);
           double cx=K(0,2);
           double cy=K(1,2);

           J <<  fx*1./z ,      0   ,  -fx*x*1./zz , -fx*xy*1./zz     , fx+(fx*xx*1./zz), -fx*y*1./z,
                 0       ,  fy*1./z ,  -fy*y*1./zz , -fy-(fy*yy)*1./zz,  fy*x*y*1./zz   ,  fy*x*1./z;
           J=-1*J;
           // cout<<"J: "<<endl<<J<<endl;
           H += J.transpose() * J;
           b += -J.transpose() * e;

           // cout<<"error_norm is : "<<error_norm<<endl;
           cost = cost + 1./2 *(e[0]*e[0]+e[1]*e[1]);
           // cout<<"cost on line is : "<<cost<<endl;
       }
//       std::cout<<"cost is :"<<cost<<std::endl;
       // while(1) {}
       // solve dx
       Vector6d dx;

       // START YOUR CODE HERE
       dx=H.ldlt().solve(b);    //the caculation of H*dx=b is right
       // cout<<"dx: "<<endl<<dx<<endl;
       // cout<<"H:"  <<endl<<H<<endl;
       // cout<<"H*dx: "<<endl<<H*dx<<endl;
       // cout<<"J^T *error =b :"<<endl<<b<<endl;
       // while(1) {}
       // END YOUR CODE HERE

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
       // START YOUR CODE HERE
       T_esti=Sophus::SE3::exp(dx)*T_esti;
       // cout<<"the new T_esti: "<<T_esti<<endl;
       // END YOUR CODE HERE

       lastCost = cost;

       // std::cout << "iteration " << iter << " cost=" << std::cout.precision(12) << cost << std::endl;
       // std::cout << "estimated pose: \n" << T_esti.matrix() << std::endl;

       }
return T_esti;
//       std::cout << "estimated pose: \n" << T_esti.matrix() << std::endl;
}
