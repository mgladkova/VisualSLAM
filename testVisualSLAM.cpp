#include "VisualSLAM.h"
#include <opencv2/viz.hpp>

void visualization_pose(std::vector<Sophus::SE3d> POSE ){
    cv::viz::Viz3d vis("Visual Odometry");
    cv::viz::WCoordinateSystem world_coor(1.0), camera_coor(0.5);
    cv::Point3d cam_pos( 0, -1.0, -1.0 ), cam_focal_point(0,0,0), cam_y_dir(0,1,0);
    cv::Affine3d cam_pose = cv::viz::makeCameraPose( cam_pos, cam_focal_point, cam_y_dir );
    vis.setViewerPose( cam_pose );

    world_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
    camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0);
    vis.showWidget( "World", world_coor );
    vis.showWidget( "Camera", camera_coor );
    for ( int i=0; i<POSE.size(); i++ )
    {
        Sophus::SE3d Tcw = POSE[i].inverse();

        // show the map and the camera pose
        cv::Mat R, t;
        cv::eigen2cv(Tcw.so3().matrix(), R);
        cv::eigen2cv(Tcw.translation(), t);
       cv::Affine3d M(R,t);

        vis.setWidgetPose( "Camera", M);
        vis.spinOnce(300, false);      // time of each picture  it is better more than 300
      }

}

int main(int argc, char** argv){

	if (argc < 5){
		std::cout << "Usage: ./slam <input_left_image_directory> <input_right_image_directory> <camera_intrinsics_file_path> <num_images>" << std::endl;
		exit(1);
	}

	std::string input_left_images_path = argv[1];
	std::string input_right_images_path = argv[2];
	std::string camera_intrinsics_path = argv[3];
	int num_images = std::stoi(argv[4]);
	std::string image_name_template = "00000";

    if (num_images <= 0)
    {
		throw std::runtime_error("The number of image pairs is invalid");
	}

	VisualSLAM slam;
    int k = 1;
	slam.readCameraIntrisics(camera_intrinsics_path);
    for (int i = 0; i < num_images; i++){
        if (i == std::pow(10, k)){
            image_name_template = image_name_template.substr(0, image_name_template.length() - 1);
            k++;
        }

        std::string image_left_name = input_left_images_path + image_name_template + std::to_string(i) + ".png";
		std::string image_right_name = input_right_images_path + image_name_template + std::to_string(i) + ".png";
        cv::Mat image_left = cv::imread(image_left_name, 0);
        cv::Mat image_right = cv::imread(image_right_name, 0);

		if (image_left.cols == 0 || image_left.rows == 0){
			throw std::runtime_error("Cannot read the image with the path: " + image_left_name);
		}

		if (image_right.cols == 0 || image_right.rows == 0){
			throw std::runtime_error("Cannot read the image with the path: " + image_right_name);
		}
        //cv::imshow("Image_left", image_left);
        //cv::imshow("Image_right", image_right);
        //cv::waitKey(0);
        slam.performFrontEndStep(image_left, image_right);
    }

    slam.visualizeAllPoses();

    /*
    std::vector<Sophus::SE3d> poses;
    for (int i = 0; i < slam.getNumberPoses(); i++){
        poses.push_back(slam.getPose(i));
    }

    visualization_pose(poses);
    */
    return 0;
}
