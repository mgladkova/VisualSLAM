#include "BundleAdjuster.h"
//#include "PoseGraph3dErrorTerm.h"

BundleAdjuster::BundleAdjuster() {}

void BundleAdjuster::prepareDataForBA(Map& map, int startFrame, int currentCameraIndex, int keyFrameStep, std::set<int> pointIndices, double* points3D, double* camera){
    std::vector<cv::Point3f> structure3D = map.getStructure3D();

    if (structure3D.empty()){
        throw std::runtime_error("prepareDataForBA() : No structure data is stored!");
    }

    std::vector<Sophus::SE3d> cameraCumPoses = map.getCumPoses();

    Sophus::SE3d firstCameraCumPose = cameraCumPoses[startFrame];

    for (int i = startFrame; i < currentCameraIndex; i += keyFrameStep){
        //Sophus::SE3d cameraPose = cameraCumPoses[i]*firstCameraCumPose.inverse();
        Sophus::SE3d cameraPose = cameraCumPoses[i];

        Eigen::Matrix3d rotation = cameraPose.so3().matrix();
        Eigen::Vector3d t = cameraPose.translation();
        Eigen::Quaterniond q(rotation);

        int cameraIndex = (i - startFrame) / keyFrameStep;

        camera[7*cameraIndex] = q.w();
        camera[7*cameraIndex + 1] = q.x();
        camera[7*cameraIndex + 2] = q.y();
        camera[7*cameraIndex + 3] = q.z();
        camera[7*cameraIndex + 4] = t[0];
        camera[7*cameraIndex + 5] = t[1];
        camera[7*cameraIndex + 6] = t[2];
    }

    int k = 0;

    for (auto it = pointIndices.begin(); it != pointIndices.end(); it++){
        if ((*it) < 0 || (*it) >= structure3D.size()){
            throw std::runtime_error("prepareDataForBA() : Index for 3D point is out of bounds!");
        }

        Eigen::Vector3d pointRelPose(structure3D[*it].x, structure3D[*it].y, structure3D[*it].z);
        //pointRelPose = firstCameraCumPose*pointRelPose;

        points3D[3*k] = pointRelPose[0];
        points3D[3*k + 1] = pointRelPose[1];
        points3D[3*k + 2] = pointRelPose[2];

        //std::cout << "POINT " << *it << " " << pointRelPose[0] << " " << pointRelPose[1] << " " << pointRelPose[2] << " " << std::endl;
        k++;
    }
}

bool BundleAdjuster::performBAWithKeyFrames(Map& map_left, Map& map_right, int keyFrameStep, int numKeyFrames){
    int currentCameraIndex = map_left.getCurrentCameraIndex();
    if (currentCameraIndex != map_right.getCurrentCameraIndex()){
        throw std::runtime_error("performBAWithKeyFrames() : unequal number of camera pose estimations for left and right cameras");
    }

    int startFrame = currentCameraIndex - keyFrameStep*numKeyFrames;

    if (startFrame < 0){
        throw std::runtime_error("prepareDataForBA() : Start frame index out of bounds!");
    }

    std::set<int> uniquePointIndices_left, uniquePointIndices_right;
    std::map<int, std::vector<std::pair<int, cv::Point2f>>> observations_left = map_left.getObservations();
    std::map<int, std::vector<std::pair<int, cv::Point2f>>> observations_right = map_right.getObservations();

    //std::cout << "START FRAME: " << startFrame << std::endl;

    for (int i = startFrame; i < currentCameraIndex; i+= keyFrameStep){
        for (int j = 0; j < observations_left[i].size(); j++){
            uniquePointIndices_left.insert(observations_left[i][j].first);
        }
    }

    for (int i = startFrame; i < currentCameraIndex; i+= keyFrameStep){
        for (int j = 0; j < observations_right[i].size(); j++){
            uniquePointIndices_right.insert(observations_right[i][j].first);
        }
    }

    double points3DArray_left[3*uniquePointIndices_left.size()];
    double cameraPose_left[7*numKeyFrames];
    double points3DArray_right[3*uniquePointIndices_right.size()];
    double cameraPose_right[7*numKeyFrames];

    prepareDataForBA(map_left, startFrame, currentCameraIndex, keyFrameStep, uniquePointIndices_left, points3DArray_left, cameraPose_left);
    prepareDataForBA(map_right, startFrame, currentCameraIndex, keyFrameStep, uniquePointIndices_right, points3DArray_right, cameraPose_right);

    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(2.0);
    double baseline = 0.53716;
    double confid = 1e3;
    Eigen::Matrix<double, 7, 7> information_matrix = Eigen::MatrixXd::Identity(7, 7)*confid;

    for (int i = startFrame; i < currentCameraIndex; i += keyFrameStep){
        for (int j = 0; j < observations_left[i].size(); j++){
            Pose3d constraint;
            constraint.p = Eigen::Vector3d(baseline, 0, 0);
            constraint.q = Eigen::Quaterniond(0,0,0,0);

            ceres::CostFunction* cost_fun = ReprojectionError::Create(observations_left[i][j].second.x, observations_left[i][j].second.y,
                                                                      constraint, information_matrix);
            int pointIndex = std::distance(uniquePointIndices_left.begin(), uniquePointIndices_left.find(observations_left[i][j].first));
            int cameraIndex = (i - startFrame) / keyFrameStep;

            problem.AddResidualBlock(cost_fun, loss_function, &(cameraPose_left[7*cameraIndex]), &(cameraPose_right[7*cameraIndex]),
                                                              &(points3DArray_left[3*pointIndex]));
        }

        for (int j = 0; j < observations_right[i].size(); j++){
            Pose3d constraint;
            constraint.p = Eigen::Vector3d(-baseline, 0, 0);
            constraint.q = Eigen::Quaterniond(0,0,0,0);

            ceres::CostFunction* cost_fun = ReprojectionError::Create(observations_right[i][j].second.x, observations_right[i][j].second.y,
                                                                      constraint, information_matrix);
            int pointIndex = std::distance(uniquePointIndices_right.begin(), uniquePointIndices_right.find(observations_right[i][j].first));
            int cameraIndex = (i - startFrame) / keyFrameStep;

            problem.AddResidualBlock(cost_fun, loss_function, &(cameraPose_right[7*cameraIndex]), &(cameraPose_left[7*cameraIndex]),
                                                              &(points3DArray_right[3*pointIndex]));
        }
    }

    problem.SetParameterBlockConstant(cameraPose_left);
    problem.SetParameterBlockConstant(cameraPose_right);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 200;
    options.function_tolerance = 1e-5;
    options.dense_linear_algebra_library_type = ceres::LAPACK;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    if (summary.IsSolutionUsable()){
        Sophus::SE3d firstFrame_left = map_left.getCumPoseAt(startFrame);
        map_left.updatePoints3D(uniquePointIndices_left, points3DArray_left, firstFrame_left);

        Sophus::SE3d firstFrame_right = map_right.getCumPoseAt(startFrame);
        map_right.updatePoints3D(uniquePointIndices_right, points3DArray_right, firstFrame_right);

        for (int i = startFrame; i < currentCameraIndex; i+=keyFrameStep){
            int cameraIndex = (i - startFrame) / keyFrameStep;
            Eigen::Quaterniond q_left(cameraPose_left[7*cameraIndex], cameraPose_left[7*cameraIndex + 1], cameraPose_left[7*cameraIndex + 2], cameraPose_left[7*cameraIndex + 3]);
            Eigen::Vector3d t_left(cameraPose_left[7*cameraIndex + 4], cameraPose_left[7*cameraIndex + 5], cameraPose_left[7*cameraIndex + 6]);

            Sophus::SE3d newPose_left(q_left.normalized().toRotationMatrix(), t_left);

            Eigen::Quaterniond q_right(cameraPose_right[7*cameraIndex], cameraPose_right[7*cameraIndex + 1], cameraPose_right[7*cameraIndex + 2], cameraPose_right[7*cameraIndex + 3]);
            Eigen::Vector3d t_right(cameraPose_right[7*cameraIndex + 4], cameraPose_right[7*cameraIndex + 5], cameraPose_right[7*cameraIndex + 6]);

            Sophus::SE3d newPose_right(q_right.normalized().toRotationMatrix(), t_right);
            //newPose = newPose*firstFrame;
            //std::cout << "Old pose " << i << " : " << map.getCumPoseAt(i).matrix() << std::endl;

            int MAX_POSE_NORM = 100;
            if (newPose_left.log().norm() <= MAX_POSE_NORM){
                map_left.setCameraPose(i, newPose_left);
            }

            if (newPose_right.log().norm() <= MAX_POSE_NORM){
                map_right.setCameraPose(i, newPose_right);
            }

            //std::cout << "New pose " << i << " : " << map.getCumPoseAt(i).matrix()  << std::endl;

        }
    }

    return summary.IsSolutionUsable();
}

/*bool BundleAdjuster::performPoseGraphOptimization(Map& map_left, Map& map_right, int keyFrameStep, int numKeyFrames) {
    int currentCameraIndex = map_left.getCurrentCameraIndex();
    int startFrame = currentCameraIndex - keyFrameStep*numKeyFrames;

    if (startFrame < 0){
        throw std::runtime_error("prepareDataForBA() : Start frame index out of bounds!");
    }
    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization* quaternion_local_parameterization =
        new ceres::EigenQuaternionParameterization;

    double confid_stereo = 1e1, confid_seq = 1e1;
    Eigen::Matrix<double, 6, 6> information_matrix_stereo = Eigen::MatrixXd::Identity(6, 6)*confid_stereo;
    Eigen::Matrix<double, 6, 6> information_matrix_seq = Eigen::MatrixXd::Identity(6, 6)*confid_seq;

    double baseline = 0.53716;

    std::vector<Sophus::SE3d> posesLeftImage = map_left.getCumPoses();
    std::vector<Sophus::SE3d> posesRightImage = map_right.getCumPoses();


    for (int i = startFrame; i < currentCameraIndex; i+= keyFrameStep) {

        if (i < 0 || i >= posesLeftImage.size() || i >= posesRightImage.size()){
            throw std::runtime_error("performPoseGraphOptimization() : Pose index out of bounds");
        }

        Pose3d constraint;
        constraint.p = Eigen::Vector3d(baseline, 0, 0);
        constraint.q = Eigen::Quaterniond(0,0,0,0);

        ceres::CostFunction* cost_function =
                PoseGraph3dErrorTerm::Create(constraint, information_matrix_stereo);

        problem.AddResidualBlock(cost_function, loss_function,
                                  posesLeftImage[i].translation().data(),
                                  posesLeftImage[i].so3().unit_quaternion().matrix().data(),
                                  posesRightImage[i].translation().data(),
                                  posesRightImage[i].so3().unit_quaternion().matrix().data());

        problem.SetParameterization(posesLeftImage[i].so3().unit_quaternion().matrix().data(),
                                     quaternion_local_parameterization);
        problem.SetParameterization(posesRightImage[i].so3().unit_quaternion().matrix().data(),
                                     quaternion_local_parameterization);
    }

    for (int i = startFrame; i < currentCameraIndex - keyFrameStep; i+= keyFrameStep){
        Sophus::SE3d relPose = posesLeftImage[i].inverse()*posesLeftImage[i + keyFrameStep];
        Eigen::Quaterniond q = relPose.unit_quaternion();
        Pose3d constraint;
        constraint.p = relPose.translation();
        constraint.q = q;


        ceres::CostFunction* cost_function =
                PoseGraph3dErrorTerm::Create(constraint, information_matrix_seq);

        problem.AddResidualBlock(cost_function, loss_function,
                                  posesLeftImage[i].translation().data(),
                                  posesLeftImage[i].so3().unit_quaternion().matrix().data(),
                                  posesLeftImage[i + keyFrameStep].translation().data(),
                                  posesLeftImage[i + keyFrameStep].so3().unit_quaternion().matrix().data());
    }


    for (int i = startFrame; i < currentCameraIndex - keyFrameStep; i+= keyFrameStep){
        Sophus::SE3d relPose = posesRightImage[i].inverse()*posesRightImage[i + keyFrameStep];
        Eigen::Quaterniond q = relPose.unit_quaternion();
        Pose3d constraint;
        constraint.p = relPose.translation();
        constraint.q = q;


        ceres::CostFunction* cost_function =
                PoseGraph3dErrorTerm::Create(constraint, information_matrix_seq);

        problem.AddResidualBlock(cost_function, loss_function,
                                  posesRightImage[i].translation().data(),
                                  posesRightImage[i].so3().unit_quaternion().matrix().data(),
                                  posesRightImage[i + keyFrameStep].translation().data(),
                                  posesRightImage[i + keyFrameStep].so3().unit_quaternion().matrix().data());
    }

    problem.SetParameterBlockConstant(posesLeftImage[startFrame].translation().data());
    problem.SetParameterBlockConstant(posesLeftImage[startFrame].so3().unit_quaternion().matrix().data());

    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << std::endl;

    int MAX_POSE_NORM = 100;

    if (summary.IsSolutionUsable()){
        for (int i = startFrame; i < currentCameraIndex; i+= keyFrameStep){

            if (posesLeftImage[i].log().norm() <= MAX_POSE_NORM){
                map_left.setCameraPose(i, posesLeftImage[i]);
            }

            if (posesRightImage[i].log().norm() <= MAX_POSE_NORM){
                map_right.setCameraPose(i, posesRightImage[i]);
            }
        }
    }

    return summary.IsSolutionUsable();
}*/
