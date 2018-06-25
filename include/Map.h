#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <vector>
#include <map>
#include <set>

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
    int offsetIndex;
public:
	Map();
    void addPoints3D(std::vector<cv::Point3f> points3D);
    void addObservations(std::vector<int> indices, std::vector<cv::Point2f> observedPoints, bool newBatch);
    void updateCumulativePose(Sophus::SE3d newTransform);

    std::vector<cv::Point3f> getStructure3D() const;
    std::map<int, std::vector<std::pair<int, cv::Point2f>>> getObservations() const;
    std::vector<Sophus::SE3d> getCumPoses() const;

    int getCurrentCameraIndex() const;
    void updateCameraIndex();

    void writeBAFile(std::string fileName, int keyFrameStep, int numCameras);
};
