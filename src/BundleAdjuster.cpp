#include "BundleAdjuster.h"

BundleAdjuster::BundleAdjuster() {}

void BundleAdjuster::prepareDataForBA(Map map, int startFrame, int currentCameraIndex, int keyFrameStep, std::set<int> pointIndices, double* points3D, double* camera){
    std::vector<cv::Point3f> structure3D = map.getStructure3D();

    if (structure3D.empty()){
        throw std::runtime_error("prepareDataForBA() : No structure data is stored!");
    }

    std::vector<Sophus::SE3d> cameraCumPoses = map.getCumPoses();

    /*for (int i = startFrame; i < currentCameraIndex; i += keyFrameStep){
        Sophus::SE3d cameraPose = cameraCumPoses[i];
        Eigen::Quaterniond q(cameraPose.so3().matrix());
        Eigen::Vector3d t = cameraPose.translation();
        int cameraIndex = (i - startFrame) / keyFrameStep;
        cameraRotations[4*cameraIndex] = q.w();
        cameraRotations[4*cameraIndex + 1] = q.x();
        cameraRotations[4*cameraIndex + 2] = q.y();
        cameraRotations[4*cameraIndex + 3] = q.z();
        cameraTranslations[3*cameraIndex] = t[0];
        cameraTranslations[3*cameraIndex + 1] = t[1];
        cameraTranslations[3*cameraIndex + 2] = t[2];
    }*/

    for (int i = startFrame; i < currentCameraIndex; i += keyFrameStep){
        Eigen::Matrix3d rotation = cameraCumPoses[i].so3().matrix();
        Eigen::Vector3d t = cameraCumPoses[i].translation();

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
        points3D[3*k] = structure3D[*it].x;
        points3D[3*k + 1] = structure3D[*it].y;
        points3D[3*k + 2] = structure3D[*it].z;
        k++;
    }
}

void BundleAdjuster::optimizeCameraPosesForKeyframes(Map map, int keyFrameStep, int numKeyFrames){
    int currentCameraIndex = map.getCurrentCameraIndex();
    int startFrame = currentCameraIndex - keyFrameStep*numKeyFrames;

    if (startFrame < 0){
        throw std::runtime_error("prepareDataForBA() : Start frame index out of bounds!");
    }

    std::set<int> uniquePointIndices;
    std::map<int, std::vector<std::pair<int, cv::Point2f>>> observations = map.getObservations();

    for (int i = startFrame; i < currentCameraIndex; i+= keyFrameStep){
        for (int j = 0; j < observations[i].size(); j++){
            uniquePointIndices.insert(observations[i][j].first);
        }
    }

    double points3DArray[3*uniquePointIndices.size()];
    double cameraPose[7*numKeyFrames];

    prepareDataForBA(map, startFrame, currentCameraIndex, keyFrameStep, uniquePointIndices, points3DArray, cameraPose);

    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

    for (int i = startFrame; i < currentCameraIndex; i += keyFrameStep){
        for (int j = 0; j < observations[i].size(); j++){
            // x and y coordinates of each observed point are stored consecutively
            ceres::CostFunction* cost_fun = ReprojectionError::Create(observations[i][j].second.x, observations[i][j].second.y);
            int pointIndex = std::distance(uniquePointIndices.begin(), uniquePointIndices.find(observations[i][j].first));
            int cameraIndex = (i - startFrame) / keyFrameStep;
            //std::cout << cameraPose + 7*cameraIndex << " " << cameraPose[7*cameraIndex] << std::endl;
            //std::cout << points3DArray + 3*pointIndex << " " << points3DArray[3*pointIndex] << std::endl;

            problem.AddResidualBlock(cost_fun, loss_function, &(cameraPose[7*cameraIndex]), &(points3DArray[3*pointIndex]));
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.gradient_tolerance = 1e-16;
    options.function_tolerance = 1e-16;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    // TO-DO: store new data
}
