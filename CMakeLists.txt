cmake_minimum_required(VERSION 3.24)
project(LidarManagerV2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options de build
option(BUILD_WITH_CUDA "Enable CUDA support" OFF)
option(BUILD_WITH_TENSORRT "Enable TensorRT" OFF)
option(BUILD_SECURITY_FEATURES "Enable security features" OFF)
option(BUILD_MONITORING "Enable monitoring" OFF)
option(ENABLE_LTO "Enable Link Time Optimization" ON)

include(cmake/DetectPlatform.cmake)

# Optimisations compilateur
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(ENABLE_LTO)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    endif()
    add_compile_options(-O3 -march=native -mtune=native)
    add_compile_options(-ffast-math -funroll-loops)
endif()

# Conan integration
include(cmake/conan_setup.cmake)

find_package(fmt REQUIRED)
find_package(OusterSDK REQUIRED)
find_package(GTest REQUIRED)


add_subdirectory(src)
add_subdirectory(tests)
