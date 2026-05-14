cmake_minimum_required(VERSION 3.10)
project(filter_compare)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(filter_compare
    filter_compare_1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../utilities/PolarChannel.cpp
)

target_include_directories(filter_compare PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../utilities)
target_link_libraries(filter_compare ${OpenCV_LIBS})

# ---- Table 2: Quantitative evaluation (GFTT + LK matching) ----
add_executable(filter_compare_table2
    filter_compare_table2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../utilities/PolarChannel.cpp
)

target_include_directories(filter_compare_table2 PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../utilities)
target_link_libraries(filter_compare_table2 ${OpenCV_LIBS})
