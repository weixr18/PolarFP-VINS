cmake_minimum_required(VERSION 2.8.3)
project(polarfp_vins)

set(PROJECT_PREFIX polarfp_vins)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

# Optional LibTorch for SuperPoint detector
option(USE_SUPERPOINT "Enable SuperPoint detector (requires LibTorch)" ON)
if(USE_SUPERPOINT)
    set(LIBTORCH_DIR "/home/dhz/SLAM/libs/libtorch-cxx11-abi-shared-with-deps-1.12.1+cu113/libtorch")
    message(STATUS "LibTorch path: ${LIBTORCH_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")
    find_package(Torch REQUIRED)
    add_definitions(-DUSE_SUPERPOINT)
    message(STATUS "SuperPoint detector: ENABLED")
endif()

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    polarfp_camera_models
    image_transport)

find_package(OpenCV REQUIRED)

# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

find_package(Ceres REQUIRED)

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

catkin_package()

add_library(vins_lib
    src/estimator/parameters.cpp
    src/estimator/estimator.cpp
    src/estimator/feature_manager.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projectionTwoFrameOneCamFactor.cpp
    src/factor/projectionTwoFrameTwoCamFactor.cpp
    src/factor/projectionOneFrameTwoCamFactor.cpp
    src/factor/marginalization_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    src/featureTracker/feature_tracker.cpp
    src/featureTracker/feature_tracker_detector.cpp
    src/featureTracker/feature_tracker_matcher.cpp
    src/featureTracker/PolarChannel.cpp)

# Conditionally add SuperPoint source
if(USE_SUPERPOINT)
    target_sources(vins_lib PRIVATE src/featureTracker/superpoint_detector.cpp)
endif()

target_link_libraries(vins_lib ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})

# Link LibTorch if enabled
if(USE_SUPERPOINT)
    target_link_libraries(vins_lib ${TORCH_LIBRARIES})
endif()


add_executable(${PROJECT_PREFIX}_node src/rosNodeTest.cpp)
target_link_libraries(${PROJECT_PREFIX}_node vins_lib)

add_executable(test_nlm_filter src/test/test_nlm_filter.cpp)
target_link_libraries(test_nlm_filter vins_lib)

# add_executable(test_guide_filter src/test/test_guide_filter.cpp)
# target_link_libraries(test_guide_filter vins_lib)
