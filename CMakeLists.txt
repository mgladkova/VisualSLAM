cmake_minimum_required(VERSION 2.8)
project(VisualSLAM)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -march=native ") #-O3 -march=native
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# eigen
include_directories("/usr/include/eigen3")  # change this if your eigen is not installed by apt-get

# sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

## project h file
include_directories("./include")
##
add_executable(testVSLAM testVSLAM.cpp src/VisualSLAM.cpp src/VisualOdometry.cpp src/Map.cpp src/BundleAdjuster.cpp)
target_link_libraries(testVSLAM ${OpenCV_LIBS} ${Pangolin_LIBRARIES} ${Sophus_LIBRARIES} ${CERES_INCLUDE_DIRS})
