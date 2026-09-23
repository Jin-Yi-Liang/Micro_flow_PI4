cmake_minimum_required(VERSION 3.16)
project(microflow_ncnn_reference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(ncnn REQUIRED)

add_executable(mnist_ncnn_benchmark mnist_ncnn_benchmark.cpp)
target_link_libraries(mnist_ncnn_benchmark PRIVATE ncnn)
target_compile_options(mnist_ncnn_benchmark PRIVATE -O3)
