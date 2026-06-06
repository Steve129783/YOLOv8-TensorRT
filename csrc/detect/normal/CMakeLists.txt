cmake_minimum_required(VERSION 3.12)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86 89 90)

project(yolov8 LANGUAGES CXX CUDA)

add_definitions(-DTRT_10)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Function)

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1")
find_package(CUDA REQUIRED)
print_var(CUDA_LIBRARIES)
print_var(CUDA_INCLUDE_DIRS)
get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)
print_var(CUDA_LIB_DIR)

# OpenCV
set(OpenCV_DIR "E:/steve/tools/opencv/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)
print_var(OpenCV_LIBS)
print_var(OpenCV_LIBRARIES)
print_var(OpenCV_INCLUDE_DIRS)

# TensorRT
set(TensorRT_ROOT "E:/steve/tools/trt_version/TensorRT-10.16.1.11")
find_package(TensorRT REQUIRED)
print_var(TensorRT_LIBRARIES)
print_var(TensorRT_INCLUDE_DIRS)
print_var(TensorRT_LIB_DIR)


if (TensorRT_VERSION_MAJOR GREATER_EQUAL 10)
    message(STATUS "Build with -DTRT_10")
    add_definitions(-DTRT_10)
endif ()

list(APPEND ALL_INCLUDE_DIRS
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

list(APPEND ALL_LIBS
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${TensorRT_LIBRARIES}
        E:/steve/tools/trt_version/TensorRT-10.16.1.11/lib/nvinfer_plugin_10.lib
)

list(APPEND ALL_LIB_DIRS
        ${CUDA_LIB_DIR}
        ${TensorRT_LIB_DIR}
)

print_var(ALL_INCLUDE_DIRS)
print_var(ALL_LIBS)
print_var(ALL_LIB_DIRS)


add_executable(
        ${PROJECT_NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yolov8.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/common.hpp
)

add_library(${PROJECT_NAME}_lib SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/yolov8det.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/yolov8det.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/common.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yolov8.hpp
)

target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX)

target_include_directories(
        ${PROJECT_NAME}_lib
        PUBLIC
        ${ALL_INCLUDE_DIRS}
)

target_link_directories(
        ${PROJECT_NAME}_lib
        PUBLIC
        ${ALL_LIB_DIRS}
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
    ${PROJECT_NAME}_lib
    ${ALL_LIBS}
)

target_link_libraries(
        ${PROJECT_NAME}_lib
        PRIVATE
        ${ALL_LIBS}
)

if (${OpenCV_VERSION} VERSION_GREATER_EQUAL 4.7.0)
    message(STATUS "Build with -DBATCHED_NMS")
    add_definitions(-DBATCHED_NMS)
endif ()
