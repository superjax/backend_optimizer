cmake_minimum_required(VERSION 2.8.3)
project(backend_optimizer)

find_package(Eigen3 REQUIRED)

set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=gnu++11 ${OpenMP_CXX_FLAGS} -O2 -mfma -mavx")

find_package(GTSAM)
find_package(pybind11)

include_directories(
        include
        ${EIGEN_INCLUDE_DIR}
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

set_target_properties(backend_optimizer PROPERTIES
    PREFIX ""
    OUTPUT_NAME "backend_optimizer"
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY ../python
)

set_target_properties(kdtree PROPERTIES
    PREFIX ""
    OUTPUT_NAME "kdtree"
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY ../python
)