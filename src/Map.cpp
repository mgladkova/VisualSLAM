#include "Map.h"

Map::Map() {
    currentCameraIndex = 0;
    offset = 0;
    mReadyToProcess = false;
    mProcessed = true;
}

void Map::addPoints3D(std::vector<cv::Point3f> points3D){
    // convert points from camera to world coordinate system
    std::unique_lock<std::mutex> lock(mReadWriteMutex2);
    Sophus::SE3d cumPose = cumPoses[currentCameraIndex];

    std::cout << cumPose.matrix() << std::endl;

    for (int i = 0; i < points3D.size(); i++){
        Eigen::Vector3d pointWorldCoord(points3D[i].x, points3D[i].y, points3D[i].z);
        std::cout << "Camera " << currentCameraIndex << " : " << pointWorldCoord[0] << " " << pointWorldCoord[1] << " " << pointWorldCoord[2] << std::endl;
        pointWorldCoord = cumPose.inverse()*pointWorldCoord;
        std::cout << "Point ADDED " << offset + i << " : " << pointWorldCoord[0] << " " << pointWorldCoord[1] << " " << pointWorldCoord[2] << std::endl;
        points3D[i] = cv::Point3f(pointWorldCoord[0],pointWorldCoord[1],pointWorldCoord[2]);
    }

    if (structure3D.empty()){
        structure3D = points3D;
    } else {
       structure3D.insert(structure3D.end(), points3D.begin(), points3D.end());
    }

}

void Map::addObservations(std::vector<int> indices, std::vector<cv::Point2f> observedPoints){
    std::unique_lock<std::mutex> lock(mReadWriteMutex2);
    std::vector<std::pair<int, cv::Point2f>> newObservations;

    assert(indices.size() == observedPoints.size());

    for (auto i = 0; i < observedPoints.size(); i++){
        if (offset + indices[i] > structure3D.size() || offset + indices[i] < 0){
            std::cerr << "addObservations() : Index " << offset + indices[i] << " out of bounds!" << std::endl;
            throw std::runtime_error("addObservations() : Index out of bounds");
        }
        newObservations.push_back(std::make_pair(offset + indices[i], observedPoints[i]));
    }

    observations[currentCameraIndex] = newObservations;
}

void Map::updateCumulativePose(Sophus::SE3d newTransform){
    std::unique_lock<std::mutex> lock(mReadWriteMutex2);
    if (cumPoses.empty()){
        cumPoses.push_back(newTransform);
        return;
    }

    assert(currentCameraIndex == cumPoses.size());

    cumPoses.push_back(newTransform.inverse()*cumPoses[currentCameraIndex - 1]);
}

int Map::getCurrentCameraIndex() {
    std::unique_lock<std::mutex> lock(mReadWriteMutex2);
    return currentCameraIndex;
}

std::vector<cv::Point3f> Map::getStructure3D() {
    std::unique_lock<std::mutex> lock(mReadWriteMutex2);
    return structure3D;
}

std::map<int, std::vector<std::pair<int, cv::Point2f>>> Map::getObservations() {
    std::unique_lock<std::mutex> lock(mReadWriteMutex2);
    return observations;
}

std::vector<Sophus::SE3d> Map::getCumPoses() {
    std::unique_lock<std::mutex> lock(mReadWriteMutex2);
    return cumPoses;
}

Sophus::SE3d Map::getCumPoseAt(int index) {
    std::unique_lock<std::mutex> lock(mReadWriteMutex2);
    if (index < 0 || index > cumPoses.size()){
        throw std::runtime_error("getCumPoseAt() : Index out of bounds!");
    }

    return cumPoses[index];
}

void Map::setCameraPose(const int i, const Sophus::SE3d newPose){
    std::unique_lock<std::mutex> lock(mReadWriteMutex2);
    if (i < 0 || i >= cumPoses.size()){
        throw std::runtime_error("setCameraPose() : Index is out of bounds!");
    }
    cumPoses[i].so3().matrix() = newPose.so3().matrix();
    cumPoses[i].translation() = newPose.translation();
}

void Map::updatePoints3D(std::set<int> uniquePointIndices, double* points3DArray, Sophus::SE3d firstCamera){
    std::unique_lock<std::mutex> lock(mReadWriteMutex2);
    if (uniquePointIndices.empty()){
        throw std::runtime_error("updatePoints3D() : No point indices are given");
    }

    int k = 0;
    for (auto i : uniquePointIndices){
        if (i < 0 || i > structure3D.size()){
            throw std::runtime_error("updatePoints3D() : index out of bounds!");
        }

        Eigen::Vector3d v(points3DArray[3*k], points3DArray[3*k + 1], points3DArray[3*k + 2]);
        //v = firstCamera.inverse()*v;
        if (v[2] > 0){
            structure3D[i] = cv::Point3f(v[0], v[1], v[2]);
        }
        k++;
    }
}

void Map::updateCameraIndex(){
    std::unique_lock<std::mutex> lock(mReadWriteMutex2);
    currentCameraIndex = currentCameraIndex + 1;
}

void Map::getDataForDrawing(int& cameraIndex,
                            Sophus::SE3d& camera,
                            std::vector<cv::Point3f>& structure3d,
                            std::vector<int>& obsIndices,
                            Sophus::SE3d& gtCamera){
    {
        std::unique_lock<std::mutex> lock(mReadWriteMutex);
        mCondVar.wait(lock, [this]{return mReadyToProcess;});

        cameraIndex = this->currentCameraIndex - 1;

		camera = this->getCumPoseAt(cameraIndex);
		structure3d = this->getStructure3D();

		std::map<int, std::vector<std::pair<int, cv::Point2f>>> observations = this->observations;

		obsIndices.clear();

		for (int i = 0; i < observations[cameraIndex].size(); i++){
		   obsIndices.push_back(observations[cameraIndex][i].first);
		}

		gtCamera = getCumPoseAt(cameraIndex); // TO-DO change to GT data
        mProcessed = true;
        //mReadyToProcess = false;
    }

    mCondVar.notify_one();
}

void Map::updateDataCurrentFrame(Sophus::SE3d pose,
                                 std::vector<cv::Point2f> trackedCurrFramePoints,
                                 std::vector<int> trackedPointIndices,
                                 std::vector<cv::Point3f> points3DCurrentFrame,
                                 bool addPoints, bool rightCamera){
    {
        std::unique_lock<std::mutex> lock(mReadWriteMutex);
        mCondVar.wait(lock, [this]{return mProcessed;});

        std::cout << "Updating camera " << currentCameraIndex << std::endl;

        updateCumulativePose(pose);
        if (addPoints){
            offset = structure3D.size();
            std::cout << "OFFSET " << offset << std::endl;
            addPoints3D(points3DCurrentFrame);
        }

        for (int i = 0; i < trackedPointIndices.size(); i++){
            Eigen::Vector3d p(structure3D[i].x, structure3D[i].y, structure3D[i].z);

            p = cumPoses[currentCameraIndex]*p;

            if (p[2] <= 0){
                trackedPointIndices.erase(trackedPointIndices.begin() + i);
                trackedCurrFramePoints.erase(trackedCurrFramePoints.begin() + i);
            }
        }

        addObservations(trackedPointIndices, trackedCurrFramePoints);

        updateCameraIndex();
	if (rightCamera){    	
		mReadyToProcess = true;
	} else {
		mReadyToProcess = false;	
	}
        //mProcessed = false;
    }
    
	mCondVar.notify_one();
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
    file << "CAMERAS" << std::endl;
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
    file << "POINTS 3D" << std::endl;
    for (auto it = uniquePointIndices.begin(); it != uniquePointIndices.end(); it++){
        int index = *it;
        if (index >= 0 && index < structure3D.size()){
            file << index << " " << float(structure3D[index].x) << " " << float(structure3D[index].y) << " " << float(structure3D[index].z) << std::endl;
        }
    }

    file.close();
}
