cmake_minimum_required(VERSION 2.8.3)
project(backend_optimizer)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
)

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=gnu++11 ${OpenMP_CXX_FLAGS} -O3")

# Add GTSAM
find_package(GTSAM)

# Get support for boost python
find_package(Boost REQUIRED COMPONENTS python)
find_package(PythonLibs 2.7 REQUIRED)
catkin_python_setup()

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES backend_optimizer _backend_optimizer_wrapper_cpp
    CATKIN_DEPENDS roscpp
    DEPENDS Boost
)

###########
## Build ##
###########
include_directories(include)
include_directories(
        ${catkin_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
        ${GTSAM_INCLUDE_DIR}
)
## Declare cpp libraries
add_library(backend_optimizer
  src/backend_optimizer.cpp
  include/backend_optimizer.h
)

add_library(_backend_optimizer_wrapper_cpp
    src/backend_optimizer_wrapper.cpp
    include/backend_optimizer.h
)

## Specify libraries to link a library or executable target against
target_link_libraries(backend_optimizer ${catkin_LIBRARIES} gtsam)
target_link_libraries(_backend_optimizer_wrapper_cpp backend_optimizer ${catkin_LIBRARIES} ${Boost_LIBRARIES})

# Don't prepend wrapper library name with lib and add to Python libs.
set_target_properties(_backend_optimizer_wrapper_cpp PROPERTIES
        PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}
        )
