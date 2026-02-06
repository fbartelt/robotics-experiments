cmake_minimum_required(VERSION 3.14)
project(adaptive_cpp LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

# # Add uaibot cpp project
# #/home/fbartelt/Documents/Projetos/UAIbotPy/uaibot/c_implementation
# # === UAIBOT C++ sources ===
# set(UAIBOT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../UAIbotPy/uaibot/c_implementation)
# set(UAIBOT_SOURCES
#     ${UAIBOT_SRC_DIR}/manipulator.cpp
#     ${UAIBOT_SRC_DIR}/utils.cpp
#     ${UAIBOT_SRC_DIR}/gjk.c
# )
#
# target_include_directories(adaptive_cpp PRIVATE
#     ${CMAKE_SOURCE_DIR}
#     ${CMAKE_SOURCE_DIR}/../UAIbotPy/uaibot/c_implementation  # headers for uaibot_cpp
# )
#

set(UAIBOT_DIR ${CMAKE_SOURCE_DIR}/../UAIbotPy/uaibot/c_implementation)
include_directories(${UAIBOT_DIR} ${UAIBOT_DIR}/third_party/quadprogpp/src)

# Add system Eigen path so <Eigen/Core> works
include_directories("/usr/include/eigen3")
# Create a local C++ library from uaibot source files
add_library(uaibot_cpp STATIC
    ${UAIBOT_DIR}/manipulator.cpp
    ${UAIBOT_DIR}/newton_euler.cpp
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

pybind11_add_module(adaptive_cpp
    bind.cpp
    adaptive.cpp
)

target_link_libraries(adaptive_cpp PRIVATE
    uaibot_cpp
)

target_compile_options(adaptive_cpp PRIVATE -Wall -Wextra -O3)

set_target_properties(adaptive_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)
