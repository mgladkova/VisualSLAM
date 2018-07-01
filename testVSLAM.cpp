#include <iostream>
#include <VisualSLAM.h>


int main(int argc, char const *argv[]){

//    std::cout<<"hello my slam "<<std::endl;
    VisualSLAM slam;

//    std::cout<<"get test value from Map.h:  "<<slam.getTestValueFromMap()<<std::endl;

    if(argc < 5) {
        std::cout << "/slam  left_image_directory right_image_directory camera_intrinsics_file num_images" << std::endl;
        exit(1);
    }


    //TO DO read input left_image and right_image
    std::string left_image_path = argv[1];
    std::string right_image_path = argv[2];
    int num_images = std::stoi(argv[3]);

    if(num_images <= 0)
    {
        throw std::runtime_error("The number of images is smaller than 0 ");
    }

    std::vector<Sophus::SE3d> groundTruthPoses;

    if (argc >= 6){
        std::string ground_truth_path = argv[5];
        slam.readGroundTruthData(ground_truth_path, num_images, groundTruthPoses);
    }
    Eigen::Vector3d translGTAccumulated, translEstimAccumulated(1,1,1);
    cv::Mat window = cv::Mat::zeros(1000, 1000, CV_8UC3);
    cv::Mat prevImageLeft;
    std::vector<cv::Point2f> pointsCurrentFrame, pointsPrevFrame;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    Eigen::Matrix3d cumR = Eigen::Matrix3d::Identity();

    std::string image_name_template = "00000";
    //TO DO read camera_instrinsics_path
    std::string camera_instrinsics_path = argv[4];
    Eigen::Matrix3d K = slam.getCameraIntrinsics(camera_instrinsics_path);
    double fx = K(0,0);

    int k(1);
    std::vector<cv::Point2f> previousFrame2DPoints, currFrame2DPoints;
    cv::Mat previousImage;
    for(int i=0;i<num_images;i++) {
        if(i == std::pow(10,k)){
            image_name_template = image_name_template.substr(0,image_name_template.length()-1);
            k++;
        }

        std::string left_image_name  = left_image_path+image_name_template + std::to_string(i)+".png";
        std::string right_image_name = right_image_path+image_name_template+ std::to_string(i)+".png";
        cv::Mat leftImage = cv::imread(left_image_name,0) ;
//        cv::Mat leftImage_BGR;
//        cv::cvtColor(leftImage,leftImage_BGR,CV_GRAY2BGR);
        cv::Mat rightImage = cv::imread(right_image_name,0);
////        // show image
//       cv::namedWindow("leftImage",CV_WINDOW_AUTOSIZE);
//       cv::imshow("leftImage",leftImage);
//       cv::waitKey();
        if(leftImage.cols <= 0 || leftImage.rows <= 0){
            throw std::runtime_error("can not read leftImage and its path is : "+ left_image_name);
        }
        if(rightImage.cols <= 0 || rightImage.rows <= 0){
            throw std::runtime_error("can not read rightImage and its path is : "+ right_image_name);
        }
        Sophus::SE3d estimatedPose;
        Sophus::SE3d prePose;
        estimatedPose = slam.estimate3D2DFrontEndWithOpicalFlow(leftImage, rightImage, previousFrame2DPoints, currFrame2DPoints, previousImage,prePose);

        prePose = estimatedPose;


        //visualization
        Sophus::SE3d cumPose;
        if (i != 0){
            cumPose = slam.getPose(i);
        }
        std::cout<<"pose "<< i << std::endl<<cumPose.matrix()<<std::endl;
        if (!groundTruthPoses.empty() && i < groundTruthPoses.size()){
            if (i == 0){
                Sophus::SE3d groundTruthPrevPose = Sophus::SE3d(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0,0,0));
                slam.plotTrajectoryNextStep(window, i, translGTAccumulated, translEstimAccumulated, groundTruthPoses[i], groundTruthPrevPose, cumR, estimatedPose);
            }
            else {
                std::cout << "Frame " << i << " / " << groundTruthPoses.size() << std::endl;
                Sophus::SE3d prevCumPose = slam.getPose(i-1);
                slam.plotTrajectoryNextStep(window, i, translGTAccumulated, translEstimAccumulated, groundTruthPoses[i], groundTruthPoses[i-1], cumR, cumPose, prevCumPose);
            }
        }
    }


}