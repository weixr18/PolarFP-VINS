cmake_minimum_required(VERSION 3.10)
project(fp_compare)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# LibTorch + CUDA (same settings as polarfp_vins_estimator)
set(LIBTORCH_DIR "/home/dhz/SLAM/libs/libtorch-cxx11-abi-shared-with-deps-1.12.1+cu113/libtorch")
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.4" CACHE PATH "" FORCE)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.4/bin/nvcc" CACHE FILEPATH "" FORCE)
find_package(Torch REQUIRED)

# ---- fp_compare_1: 3x3 feature point overlay visualization ----
add_executable(fp_compare_1
    fp_compare_1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../utilities/PolarChannel.cpp
)
target_include_directories(fp_compare_1 PRIVATE
    ${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../utilities)
target_link_libraries(fp_compare_1 ${OpenCV_LIBS} ${TORCH_LIBRARIES})

# ---- fp_compare_table3: quantitative detector evaluation ----
add_executable(fp_compare_table3
    fp_compare_table3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../utilities/PolarChannel.cpp
)
target_include_directories(fp_compare_table3 PRIVATE
    ${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../utilities)
target_link_libraries(fp_compare_table3 ${OpenCV_LIBS} ${TORCH_LIBRARIES})
