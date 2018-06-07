cmake_minimum_required(VERSION 2.8)
project(VisualSLAM)
set(CMAKE_CXX_FLAGS "-std=c++11 -g -O3 -march=native ")

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

include_directories("./include")

add_executable(slam testVisualSLAM.cpp src/VisualSLAM.cpp src/Map.cpp src/VisualOdometry.cpp src/BundleAdjuster.cpp)
target_link_libraries(slam ${OpenCV_LIBS} ${CERES_LIBRARIES} ${Pangolin_LIBRARIES} ${Sophus_LIBRARIES})
