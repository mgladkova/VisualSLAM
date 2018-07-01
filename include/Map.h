#include <sophus/se3.h>
#include <vector>



class Map{
private:
    int test_a;
    std::vector<Sophus::SE3> cumPose;
    int poseIndex;

public:
    Map();
    int getValue();

    void updateCumPose(Sophus::SE3 newPose);
    void updatePoseIndex();

    std::vector<Sophus::SE3> getCumPose();
};