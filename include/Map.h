#include <sophus/se3.hpp>
#include <vector>



class Map{
private:
    int test_a;
    std::vector<Sophus::SE3d> cumPose;
    int poseIndex;

public:
    Map();
    int getValue();

    void updateCumPose(Sophus::SE3d newPose);
    void updatePoseIndex();

    std::vector<Sophus::SE3d> getCumPose();
};