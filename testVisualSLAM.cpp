#include "VisualSLAM.h"
#include "GlobalParam.h"

int main(int argc, char** argv){

	if (argc < 5){
        std::cout << "Usage: ./slam <input_left_image_directory> <input_right_image_directory> <camera_intrinsics_file_path> <num_images> [ground_truth_data_path]" << std::endl;
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
    Eigen::Vector3d translGTAccumulated, translEstimAccumulated;
    bool gtDataLoaded = false;

    if (argc >= 6){
        std::string ground_truth_path = argv[5];
        slam.readGroundTruthData(ground_truth_path, num_images);
        gtDataLoaded = true;
    }

    cv::Mat window = cv::Mat::zeros(500, 500, CV_8UC3);
    for (int i = 0; i < num_images; i++){

        std::cout << "FRAME " << i << std::endl;
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
        if (gtDataLoaded){
            if (i > 1){
                slam.plotTrajectoryNextStep(window, i - 1, translGTAccumulated,translEstimAccumulated);
                cv::imshow("GT vs Estimated position", window);
                cv::waitKey(3);
            }
        }
    }
#ifdef VIS_ALL_POSES
    slam.visualizeAllPoses();
#endif
    return 0;
}
