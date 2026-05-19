cmake_minimum_required(VERSION 3.14)
project(sdf LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

# set(fcl_DIR "${CMAKE_SOURCE_DIR}/third_party/fcl/install/share/fcl")
# find_package(fcl REQUIRED)
# Server
# set(UAIBOT_DIR ${CMAKE_SOURCE_DIR}/../../debugenv/lib/python3.13/site-packages/uaibot/c_implementation)
# Point to the FCL source include directory (where fcl/… headers live)
set(FCL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/fcl/include")
set(FCL_SRC_INCLUDE "${CMAKE_SOURCE_DIR}/third_party/fcl/include")
set(FCL_BUILD_INCLUDE "${CMAKE_SOURCE_DIR}/third_party/fcl/build/include")
# Point to the static FCL library you built
set(FCL_LIBRARY "${CMAKE_SOURCE_DIR}/third_party/fcl/build/lib/libfcl.a")
set(CCD_LIBRARY "${CMAKE_SOURCE_DIR}/third_party/libccd/build/src/libccd.a")
# Local computer
set(UAIBOT_DIR ${CMAKE_SOURCE_DIR}/../../UAIbotPy/uaibot/c_implementation)
include_directories(${UAIBOT_DIR} ${UAIBOT_DIR}/third_party/quadprogpp/src)

# Add system Eigen path so <Eigen/Core> works
include_directories("/usr/include/eigen3")
# Create a local C++ library from uaibot source files
add_library(uaibot_cpp STATIC
    ${UAIBOT_DIR}/manipulator.cpp
    ${UAIBOT_DIR}/smooth_functions.cpp
    ${UAIBOT_DIR}/utils.cpp
    ${UAIBOT_DIR}/gjk.c
)
set_source_files_properties(${UAIBOT_DIR}/gjk.c PROPERTIES LANGUAGE C)
target_include_directories(uaibot_cpp PUBLIC
    /usr/include/eigen3
    ${UAIBOT_DIR}
)
target_include_directories(uaibot_cpp SYSTEM PUBLIC
    ${UAIBOT_DIR}/third_party
)
target_compile_features(uaibot_cpp PUBLIC cxx_std_14)

pybind11_add_module(sdf
    sdf.cpp
)

target_include_directories(sdf PRIVATE 
    ${FCL_SRC_INCLUDE}
    ${FCL_BUILD_INCLUDE}
)
target_link_libraries(sdf PRIVATE
    uaibot_cpp
    ${FCL_LIBRARY}
    ${CCD_LIBRARY}
)

target_compile_options(sdf PRIVATE -Wall -Wextra -O3)

# Assume target name is 'fcl_distance' – adjust if yours is different
set_target_properties(sdf PROPERTIES
    BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/fcl/install/lib"
    INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/fcl/install/lib"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_target_properties(sdf PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)
