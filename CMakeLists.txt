cmake_minimum_required(VERSION 3.20)
project(nToken CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(nToken main.cu)

set_target_properties(nToken PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
