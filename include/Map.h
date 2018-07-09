#pragma once

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <vector>
#include <map>
#include <set>
#include <mutex>
#include <condition_variable>

#include "VisualizationToolkit.h"

/**
 * Class Map is responsible for storing the 3D structure
 */
class Map {
private:
    std::vector<Sophus::SE3d> cumPoses;
    std::map<int, std::vector<std::pair<int, cv::Point2f>>> observations;
    std::vector<cv::Point3f> structure3D;
    int currentCameraIndex;
    std::mutex mReadWriteMutex;
    std::condition_variable mCondVar;

    bool mReadyToProcess;
    bool mProcessed;

    int offset;

public:
	Map();
    void addPoints3D(std::vector<cv::Point3f> points3D);
    void addObservations(std::vector<int> indices, std::vector<cv::Point2f> observedPoints);
    void updateCumulativePose(Sophus::SE3d newTransform);
    void updatePoints3D(std::set<int> uniquePointIndices, double* points3DArray, Sophus::SE3d firstCamera);
    void setCameraPose(const int i, const Sophus::SE3d newPose);

    std::vector<cv::Point3f> getStructure3D() const;
    std::map<int, std::vector<std::pair<int, cv::Point2f>>> getObservations() const;
    std::vector<Sophus::SE3d> getCumPoses() const;
    Sophus::SE3d getCumPoseAt(int index) const;

    void getDataForDrawing(int& cameraIndex, Sophus::SE3d& camera, std::vector<cv::Point3f>& structure3d, std::vector<int>& obsIndices, Sophus::SE3d& gtCamera);
    void updateDataCurrentFrame(Sophus::SE3d& pose, std::vector<cv::Point2f>& trackedCurrFramePoints, std::vector<int>& trackedPointIndices, std::vector<cv::Point3f>& points3DCurrentFrame, bool addPoints);

    int getCurrentCameraIndex() const;
    void updateCameraIndex();

    void writeBAFile(std::string fileName, int keyFrameStep, int numCameras);
};
