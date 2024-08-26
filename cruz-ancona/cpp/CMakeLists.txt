cmake_minimum_required(VERSION 3.12)
project(pybind_cruzancona)

# Find Pybind11 package
find_package(pybind11 REQUIRED)

# Find Eigen Package
find_package(Eigen3 REQUIRED)

# Add the pybind11 module
pybind11_add_module(pybind_cruzancona bindings.cpp)

# Link the Eigen3 library
target_link_libraries(pybind_cruzancona PRIVATE Eigen3::Eigen)