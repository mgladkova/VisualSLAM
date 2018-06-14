# VisualSLAM
Visual-based Navigation project

## Notes:
1. Please create a feature branch for your work and use master branch only for working solutions. 
2. **No build flies, images or executables are allowed** to be commited to the repository.
3. CMake is used to build the project

## How to build and run the code:
1. Download and install VTK for opencv Viz module: https://www.vtk.org/download/
2. If you have already installed OpenCV, you need to reinstall it such that OpenCV can find freshly-installed VTK library (credits to https://stackoverflow.com/questions/23559925/how-do-i-use-install-viz-in-opencv?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa)
3.
```bash
mkdir build
cmake ..
make
./slam ../data/left ../data/right 10

## only 22 for visualization otherwise we can only see pose data
./slam ../data/left/ ../data/right/ ../data/calib.txt 22
```

## something yinglong has changed:
1. add bundle ajusterment of ceres version
2. performFrontEndStep(), estimatePose3D2D(), so that we optimize the P3d[inlier_index] and P2d[inlier_index] instead of p3d[all] and p2d[all]
3. please take care, I am using old version of Sophus library, I try to change all SE3-->SE3d and #inlclude<sophus/se3.h> -->#include <sophus/se3.hpp>
   if you complie with the error of sophus, pleas change it as I said or contact me in messenger
