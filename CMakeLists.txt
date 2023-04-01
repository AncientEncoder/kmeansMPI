cmake_minimum_required(VERSION 3.20)
project(kmeansMPI)

set(CMAKE_CXX_STANDARD 11)
find_package(MPI REQUIRED)
add_executable(kmeansMPI main.cpp IOController.hpp IOController.cpp KMeans.cpp KMeans.hpp Point.cpp Point.hpp)
target_link_libraries(kmeansMPI PUBLIC MPI::MPI_CXX)