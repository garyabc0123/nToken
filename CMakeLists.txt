cmake_minimum_required(VERSION 3.20) # set cmake version
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc) #link cuda compiler
project(nToken CUDA CXX) #project name and language
set(CMAKE_CUDA_STANDARD 17) #c++ version
#set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

set(GCC_VERSION 11) #gcc version(define by me)
include_directories(/usr/include/c++/${GCC_VERSION}/ include src)

find_package(JNI)

if (JNI_FOUND)
    message (STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
    message (STATUS "JNI_LIBRARIES=${JNI_LIBRARIES}")
endif()
include_directories(${JNI_INCLUDE_DIRS})

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


message("BUILD_FUNC=${BUILD_FUNC}")
if(BUILD_FUNC MATCHES "shared")
    add_library(nToken SHARED src/nToken.cu include/nToken.cuh include/operatorParser.cuh include/basicDataTypeDefine.cuh src/operatorParser.cu src/fileReader.cu include/fileReader.cuh src/documentParser.cu include/documentParser.cuh src/gpuStack.cu include/gpuStack.cuh src/nTokenGetPosition.cu src/nTokenDfsTraversalSearch.cu include/linkWithJava.cpp include/linkWithJava.h)
else()
    add_executable(nToken main.cu src/nToken.cu include/nToken.cuh include/operatorParser.cuh include/basicDataTypeDefine.cuh src/operatorParser.cu src/fileReader.cu include/fileReader.cuh src/documentParser.cu include/documentParser.cuh src/gpuStack.cu include/gpuStack.cuh src/nTokenGetPosition.cu src/nTokenDfsTraversalSearch.cu include/linkWithJava.cpp include/linkWithJava.h)
endif()



    set_property(TARGET nToken PROPERTY CUDA_STANDARD 17)

target_link_libraries(nToken -lpthread)

#add_subdirectory(test) #test sub directory
if(BUILD_FUNC MATCHES "shared")
    set_target_properties(nToken PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            VERSION     1
            SOVERSION   1)
else()
    set_target_properties(nToken PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            VERSION     2.5
            SOVERSION   1)
endif()
