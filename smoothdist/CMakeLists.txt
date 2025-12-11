cmake_minimum_required(VERSION 3.14)
project(smoothfunctions LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

# Add system Eigen path so <Eigen/Core> works
include_directories("/usr/include/eigen3")
pybind11_add_module(smoothfunctions
    bind.cpp
    smoothfunctions.cpp
)

target_compile_options(smoothfunctions PRIVATE -Wall -Wextra -O3)

set_target_properties(smoothfunctions PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)
