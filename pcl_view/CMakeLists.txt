cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(main)

set(PCL_DIR "/home/javier/pcl-pcl-1.8.0/build/PCLConfig.cmake")
find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (main main.cpp)
target_link_libraries (main ${PCL_LIBRARIES})

