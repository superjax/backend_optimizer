cmake_minimum_required(VERSION 2.8.3)
project(backend_optimizer)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
)

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=gnu++11 ${OpenMP_CXX_FLAGS} -O3")

find_package(GTSAM)
find_package(pybind11)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES backend_optimizer
    CATKIN_DEPENDS roscpp
)

###########
## Build ##
###########
include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${GTSAM_INCLUDE_DIR}
)
## Declare a pybind module this is a fancy wrapper for add_library
pybind11_add_module(backend_optimizer
    src/backend_optimizer.cpp
    include/backend_optimizer.h
)

pybind11_add_module(kdtree
    src/kdtree.cpp
    include/kdtree.h
    include/nanoflann.hpp
)

## Specify libraries to link a library or executable target against
target_link_libraries(backend_optimizer gtsam)


# Don't prepend wrapper library name with lib and add to Python libs.
set_target_properties(backend_optimizer PROPERTIES
        PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}
)

# Put an __init__.py file in the package location so it can be imported
file(WRITE ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}/__init__.py "")
