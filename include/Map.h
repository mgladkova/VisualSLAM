#include <Eigen/Core>
#include <vector>

/**
 * Class Map is responsible for storing the 3D structure
 */
class Map {
private:
	std::vector<Eigen::Vector3d> structure3D;

public:
	Map();
	void updateStructure3d(); // update 3D point locations
};
