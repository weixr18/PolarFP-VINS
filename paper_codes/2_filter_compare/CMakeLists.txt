cmake_minimum_required(VERSION 3.10)
project(filter_compare)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(filter_compare
    filter_compare_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../utilities/PolarChannel.cpp
)

target_include_directories(filter_compare PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../utilities)
target_link_libraries(filter_compare ${OpenCV_LIBS})
