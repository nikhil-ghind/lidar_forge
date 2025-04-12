cmake_minimum_required(VERSION 3.22)
project(synthetic_lidar_generator)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.4 REQUIRED)
find_package(nlohmann_json REQUIRED)
# PCL and Torch are optional; set LIDAR_NO_PCL=ON / LIDAR_NO_TORCH=ON to skip
option(LIDAR_NO_PCL "Build without PCL" ON)
option(LIDAR_NO_TORCH "Build without libtorch" ON)

add_subdirectory(src)
enable_testing()
add_subdirectory(tests)
