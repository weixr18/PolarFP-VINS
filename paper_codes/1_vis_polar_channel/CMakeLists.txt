cmake_minimum_required(VERSION 3.10)
project(vis_polar_channel)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(vis_polar_channel
    main.cpp
    PolarChannel.cpp
)

target_include_directories(vis_polar_channel PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vis_polar_channel ${OpenCV_LIBS})
