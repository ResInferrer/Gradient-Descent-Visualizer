cmake_minimum_required(VERSION 3.15)
project(gdv_vis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Connect the core folder, where the main C++ library is located
add_subdirectory(core)

# Connect the folder with the Python wrapper
add_subdirectory(core/bindings)