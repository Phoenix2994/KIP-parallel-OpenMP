cmake_minimum_required(VERSION 3.9)
project(kip_openmp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -fopenmp")


add_executable(kip_openmp main.cpp image.cpp image.h ppm.cpp ppm.h utils.h kernels.h)