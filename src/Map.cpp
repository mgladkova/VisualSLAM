#include "Map.h"



Map::Map() {
    test_a =666;
    poseIndex = 0 ;

}

int Map::getValue() {
    return test_a;
}

void Map::updateCumPose(Sophus::SE3d newPose) {

    if (cumPose.empty()){
        cumPose.push_back(newPose);
    }
    assert(poseIndex == cumPose.size());
    cumPose.push_back(newPose.inverse() * cumPose[poseIndex - 1]);
}

void Map::updatePoseIndex() {

    poseIndex = poseIndex + 1 ;
}

std::vector<Sophus::SE3d> Map::getCumPose() {
    return cumPose;
}