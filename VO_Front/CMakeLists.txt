cmake_minimum_required(VERSION 2.8)
project(vo1_front)

#add_definitions("-DENABLE_SSE")

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS "-O3 ${SSE_FLAGS} -msse4")

find_package(OpenCV 3.2)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 3.2 not found.")
   endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

include_directories(
        ${OpenCV_INCLUDE_DIRS}
)

add_executable(opencv_orb opencv_orb.cc)
target_link_libraries(opencv_orb ${OpenCV_LIBS})

add_executable(my_orb my_orb.cc)
target_link_libraries(my_orb ${OpenCV_LIBS})