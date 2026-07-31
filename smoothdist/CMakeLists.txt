cmake_minimum_required(VERSION 3.14)
project(sdf LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 1. Tell CMake to look for packages inside our third_party/install folder
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party/install")

find_package(pybind11 CONFIG REQUIRED)
# find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Eigen3 CONFIG REQUIRED)
find_package(fcl CONFIG REQUIRED)

set(UAIBOT_DIR ${CMAKE_SOURCE_DIR}/../../UAIbotPy/uaibot/c_implementation)
# include_directories(${UAIBOT_DIR} ${UAIBOT_DIR}/third_party/quadprogpp/src)

# Add system Eigen path so <Eigen/Core> works
# Avoids using uaibot internal eigen
include_directories("/usr/include/eigen3")
# Create a local C++ library from uaibot source files
add_library(uaibot_cpp STATIC
    ${UAIBOT_DIR}/manipulator.cpp
    ${UAIBOT_DIR}/smooth_functions.cpp
    ${UAIBOT_DIR}/utils.cpp
    ${UAIBOT_DIR}/gjk.c
)
set_source_files_properties(${UAIBOT_DIR}/gjk.c PROPERTIES LANGUAGE C)

target_link_libraries(uaibot_cpp PUBLIC Eigen3::Eigen)
target_include_directories(uaibot_cpp PUBLIC
    # /usr/include/eigen3
    ${UAIBOT_DIR}
)
target_include_directories(uaibot_cpp SYSTEM PUBLIC
    ${UAIBOT_DIR}/third_party
)
target_compile_features(uaibot_cpp PUBLIC cxx_std_14)

pybind11_add_module(sdf
    sdf.cpp
)

# fcl automatically brings in its include directories AND links to CCD.
target_link_libraries(sdf PRIVATE
    uaibot_cpp
    fcl        # No need to manually link ${CCD_LIBRARY} or ${FCL_LIBRARY}!
)

target_compile_options(sdf PRIVATE -Wall -Wextra -O3)

set_target_properties(sdf PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)
