#include "Map.h"

Map::Map() {
    currentCameraIndex = 0;
    offsetIndex = 0;
}

void Map::addPoints3D(std::vector<cv::Point3f> points3D){
    // convert points from camera to world coordinate system
    Sophus::SE3d cumPose = cumPoses[currentCameraIndex];

    for (int i = 0; i < points3D.size(); i++){
        Eigen::Vector3d pointWorldCoord(points3D[i].x, points3D[i].y, points3D[i].z);
        pointWorldCoord = cumPose.inverse()*pointWorldCoord;
        structure3D.insert(structure3D.end(), cv::Point3f(pointWorldCoord[0],pointWorldCoord[1],pointWorldCoord[2]));
    }
}

void Map::addObservations(std::vector<int> indices, std::vector<cv::Point2f> observedPoints, bool newBatch){
    std::vector<std::pair<int, cv::Point2f>> newObservations;

    assert(indices.size() == observedPoints.size());

    if (newBatch){
        offsetIndex = structure3D.size();
    }

    for (auto i = 0; i < observedPoints.size(); i++){
        newObservations.push_back(std::make_pair(offsetIndex + indices[i], observedPoints[i]));
    }

    observations[currentCameraIndex] = newObservations;
}

void Map::updateCumulativePose(Sophus::SE3d newTransform){
    if (cumPoses.empty()){
        cumPoses.push_back(newTransform);
        return;
    }

    assert(currentCameraIndex == cumPoses.size());

    cumPoses.push_back(newTransform*cumPoses[currentCameraIndex - 1]);
}

int Map::getCurrentCameraIndex() const{
    return currentCameraIndex;
}

std::vector<cv::Point3f> Map::getStructure3D() const{
    return structure3D;
}

std::map<int, std::vector<std::pair<int, cv::Point2f>>> Map::getObservations() const{
    return observations;
}

std::vector<Sophus::SE3d> Map::getCumPoses() const{
    return cumPoses;
}

void Map::updateCameraIndex(){
    currentCameraIndex = currentCameraIndex + 1;
}

void Map::writeBAFile(std::string fileName, int keyFrameStep, int numKeyFrames) {
    std::ofstream file;
    file.open(fileName, std::ofstream::out | std::ofstream::trunc);

    if (!file.is_open()){
        throw std::runtime_error("writeBAFile() : Could not open file to write!");
    }

    int startFrame = currentCameraIndex - keyFrameStep*numKeyFrames;

    if (startFrame < 0){
        throw std::runtime_error("writeBAFile() : Start frame index out of bounds!");
    }

    if (structure3D.empty() || observations.empty()){
        throw std::runtime_error("writeBAFile() : No structure data is stored!");
    }
    std::set<int> uniquePointIndices;

    size_t numObservations = 0;

    for (int i = startFrame; i < currentCameraIndex; i+= keyFrameStep){
        for (int j = 0; j < observations[i].size(); j++){
            uniquePointIndices.insert(observations[i][j].first);
        }
        numObservations += observations[i].size();
    }

    file << int(numKeyFrames) << " " << int(uniquePointIndices.size()) << " " << int(numObservations) << std::endl;

    for (int i = startFrame; i < currentCameraIndex; i += keyFrameStep){
        for (int j = 0; j < observations[i].size(); j++){
            file << int((i  - startFrame) / keyFrameStep) << " " << int(observations[i][j].first) << " " << double(observations[i][j].second.x) << " " << double(observations[i][j].second.y) << std::endl;
        }
    }

    for (int i = startFrame; i < currentCameraIndex; i += keyFrameStep){
        if (i >= cumPoses.size()){
            throw std::runtime_error("writeBAFile() : Index out of bounds for pose extraction!");
        }

        Eigen::Matrix3d rotation = cumPoses[i].so3().matrix();
        Eigen::Vector3d t = cumPoses[i].translation();

        Eigen::Quaterniond q(rotation);

        file << q.w() << std::endl;
        file << q.x() << std::endl;
        file << q.y() << std::endl;
        file << q.z() << std::endl;

        file << t[0] << std::endl;
        file << t[1] << std::endl;
        file << t[2] << std::endl;


        /*Eigen::AngleAxisd rvec;
        rvec.fromRotationMatrix(rotation);

        file << rvec.angle()*rvec.axis()[0] << std::endl;
        file << rvec.angle()*rvec.axis()[1] << std::endl;
        file << rvec.angle()*rvec.axis()[2] << std::endl;
        file << t[0] << std::endl;
        file << t[1] << std::endl;
        file << t[2] << std::endl;
        file << 718.856 << std::endl;
        file << 0.0 << std::endl;
        file << 0.0 << std::endl;*/
    }

    for (auto it = uniquePointIndices.begin(); it != uniquePointIndices.end(); it++){
        int index = *it;
        if (index >= 0 && index < structure3D.size()){
            file << double(structure3D[index].x) << std::endl;
            file << double(structure3D[index].y) << std::endl;
            file << double(structure3D[index].z) << std::endl;
        }
    }

    file.close();
}
