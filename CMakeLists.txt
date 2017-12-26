cmake_minimum_required(VERSION 3.7)
project(tf_detector_example)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp utils.cpp utils.h)
add_executable(tf_detector_example ${SOURCE_FILES})

# OpenCV libs
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(tf_detector_example ${OpenCV_LIBS})


# ==================== PATHS TO SPECIFY! ==================== #

# Eigen lib headers
include_directories("/usr/local/include/eigen3")

# TensorFlow headers
include_directories("/usr/local/include/tf/")
include_directories("/usr/local/include/tf/bazel-genfiles/")
include_directories("/usr/local/include/tf/tensorflow/")
include_directories("/usr/local/include/tf/third-party/")

# Link TensorFlow libs
target_link_libraries(tf_detector_example "/usr/local/lib/libtensorflow_cc.so")
target_link_libraries(tf_detector_example "/usr/local/lib/libtensorflow_framework.so")
