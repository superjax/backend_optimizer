cmake_minimum_required(VERSION 2.8.3)
project(backend_optimizer)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
)

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=gnu++11 ${OpenMP_CXX_FLAGS} -O3")

# Add GTSAM
 # find_package(GTSAM)

# Get support for boost python
find_package(pybind11)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES backend_optimizer
    CATKIN_DEPENDS roscpp
)

###########
## Build ##
###########
include_directories(include)
include_directories(
        ${catkin_INCLUDE_DIRS}
)
## Declare a pybind module
pybind11_add_module(backend_optimizer src/backend_optimizer.cpp)

# add_library(_backend_optimizer_wrapper_cpp
#     MODULE
#     src/backend_optimizer_wrapper.cpp
#     include/backend_optimizer.h
# )

## Specify libraries to link a library or executable target against
# target_link_libraries(backend_optimizer ${catkin_LIBRARIES} gtsam)
# target_link_libraries(_backend_optimizer_wrapper_cpp backend_optimizer ${catkin_LIBRARIES} ${Boost_LIBRARIES})

# Don't prepend wrapper library name with lib and add to Python libs.
set_target_properties(backend_optimizer PROPERTIES
        PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
