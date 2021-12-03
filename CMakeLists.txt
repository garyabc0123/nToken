cmake_minimum_required(VERSION 3.20) # set cmake version
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc) #link cuda compiler
project(nToken CUDA CXX) #project name and language
set(CMAKE_CUDA_STANDARD 20) #c++ version
#set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

set(GCC_VERSION 11) #gcc version(define by me)
if(NOT WIN32)
    include_directories(/usr/include/c++/${GCC_VERSION})
    message("include linux header")
endif()
include_directories(include)
include_directories(src)

#LINUX
if(WIN32)
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -lineinfo  --extended-lambda ")
    message("WIN32")
else()
    if(CMAKE_BUILD_TYPE MATCHES "Debug") #debug mode
        set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -lineinfo --compiler-bindir /usr/bin/g++-${GCC_VERSION} --extended-lambda ")
        message("debug mode")
    else()                               #release mode
        set(CMAKE_CUDA_FLAGS " --compiler-bindir /usr/bin/g++-${GCC_VERSION} --extended-lambda -O3 ")
        message("release mode")
    endif()
endif()

enable_testing() #enable gtest





add_executable(nToken main.cu src/nToken.cu include/nToken.cuh include/operatorParser.cuh include/basicDataTypeDefine.cuh src/operatorParser.cu src/fileReader.cu include/fileReader.cuh src/documentParser.cu include/documentParser.cuh src/gpuStack.cu include/gpuStack.cuh include/hyperCompressBoolArray.cuh src/hyperCompressBoolArray.cu)
set_property(TARGET nToken PROPERTY CUDA_STANDARD 20)

target_link_libraries(nToken -lpthread)

add_subdirectory(test) #test sub directory
set_target_properties(nToken PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
