cmake_minimum_required(VERSION 3.0.0)

project(hScanDetect VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt5Core)

find_package(Python3 COMPONENTS Development NumPy)

set(OpenCV_DIR /opt/opencv454/lib/cmake/opencv4/)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(/usr/include/eigen3)

add_executable(${PROJECT_NAME} "main.cpp")

target_link_libraries(${PROJECT_NAME} Qt5::Core)
target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} Python3::Python Python3::NumPy pthread)
