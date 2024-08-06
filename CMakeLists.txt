cmake_minimum_required(VERSION 3.10.0)

project(hScanDetect VERSION 0.1 LANGUAGES CXX)

# set(CMAKE_C_COMPILER "/bin/x86_64-linux-gnu-cpp")
# set(CMAKE_CXX_COMPILER "/bin/x86_64-linux-gnu-g++")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/")

find_package(Qt5Core)

# 检查是否找到 Qt5Core
if(Qt5Core_FOUND)
    message(STATUS "Found Qt5Core")
    message(STATUS "Qt5Core_INCLUDE_DIRS:${Qt5Core_INCLUDE_DIRS}")
    message(STATUS "QT5Core_LIBRARIES:${Qt5Core_LIBRARIES}")
else()
    message(FATAL_ERROR "Qt5Core not found")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
# 检查是否找到Python3
if(Python3_FOUND)
     message(STATUS "Found Python3: ${Python3_EXECUTABLE}")
     include_directories(${Python3_INCLUDE_DIRS})
else()
     message(FATAL_ERROR "Python3 not found")
endif()

set(OpenCV_DIR /opt/opencv454/lib/cmake/opencv4/)
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found")
    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}")
    message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "OpenCV not found")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

# 查找 Eigen3
find_package(Eigen3 REQUIRED)

# 检查是否找到 Eigen3
if(Eigen3_FOUND)
    message(STATUS "Found Eigen3: ${Eigen3_INCLUDE_DIRS}")
    include_directories(${Eigen3_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Eigen3 not found")
endif()

add_executable(${PROJECT_NAME} "main.cpp")

target_link_libraries(${PROJECT_NAME} Qt5::Core)
target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} Python3::Python Python3::NumPy pthread)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

