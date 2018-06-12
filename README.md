#1 about libirary
#### g2o use the old version(2016) instead of 2017
#### sophus use the old version, so if you are using the new version, SE3d should change to SE3

#2 about my changing
### add disparity function in test/run_vo.cpp;
### add associate.txt to let code find path of kitti dataset, the associate.txt should be in "/kitti_data_set/dataset/sequences/00/associate.txt";
### for git my associate.txt you can find in "yinglong_added_file" file
### modify config/defualt.yaml; add path to kitti dataset and add cameta camera intrinsics

#3 run the code
cd project/0.3
mkdir build
cd build
cmake ..
make

cd ..
bin/run_vo config/default.yaml

