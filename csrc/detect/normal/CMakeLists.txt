cmake_minimum_required(VERSION 3.12)
# use -T v143,version=14.38 !!!!
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(yolov8 LANGUAGES CXX CUDA)
add_definitions(-DTRT_10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Function)

# CUDA
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    if(WIN32)
        set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")
    elseif(UNIX)
        set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
    endif()
endif()
find_package(CUDA REQUIRED)
print_var(CUDA_LIBRARIES)
print_var(CUDA_INCLUDE_DIRS)
get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)
print_var(CUDA_LIB_DIR)

# OpenCV
if(NOT DEFINED OpenCV_DIR)
    if(WIN32)
        set(OpenCV_DIR "D:/tools/opencv/opencv490/build")
    elseif(UNIX)
        set(OpenCV_DIR "/home/steve/tools/opencv/install/opencv-4.9.0/lib/cmake/opencv4")
    endif()
endif()
find_package(OpenCV REQUIRED)
print_var(OpenCV_LIBS)
print_var(OpenCV_LIBRARIES)
print_var(OpenCV_INCLUDE_DIRS)

# TensorRT
if(NOT DEFINED TensorRT_ROOT)
    if(WIN32)
        set(TensorRT_ROOT "D:/tools/trt_versions/TensorRT-10.16.1.11")
    elseif(UNIX)
        set(TensorRT_ROOT "/home/steve/tools/tensorrt/TensorRT-10.16.1.11")
    endif()
endif()
find_package(TensorRT REQUIRED)
print_var(TensorRT_LIBRARIES)
print_var(TensorRT_INCLUDE_DIRS)
print_var(TensorRT_LIB_DIR)

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

add_executable(
        test
        ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
)

target_include_directories(
        test
        PRIVATE
        ${ALL_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
        test
        PRIVATE
        ${OpenCV_LIBRARIES}
)

add_library(${PROJECT_NAME}_lib SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/yolov8det.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/yolov8det.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/common.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/yolov8.hpp
)

if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX)
    target_compile_definitions(${PROJECT_NAME}_lib PRIVATE NOMINMAX)
endif()

if (TensorRT_VERSION_MAJOR GREATER_EQUAL 10)
    message(STATUS "Build with -DTRT_10")
    target_compile_definitions(${PROJECT_NAME} PRIVATE TRT_10)
    target_compile_definitions(${PROJECT_NAME}_lib PRIVATE TRT_10)
endif()

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
    target_compile_definitions(${PROJECT_NAME} PRIVATE BATCHED_NMS)
    target_compile_definitions(${PROJECT_NAME}_lib PRIVATE BATCHED_NMS)
endif()

install(TARGETS yolov8 yolov8_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)