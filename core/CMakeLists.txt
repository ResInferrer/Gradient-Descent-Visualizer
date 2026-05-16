cmake_minimum_required(VERSION 3.15)
project(gdv_core LANGUAGES CXX)

# Take all .cpp files from src and all .h files from include
file(GLOB_RECURSE GDV_SOURCES "src/*.cpp")
file(GLOB_RECURSE GDV_HEADERS "include/*.h")

# Create a static library (build speed + convenience)
add_library(gdv_core STATIC ${GDV_SOURCES} ${GDV_HEADERS})

# Add the PIC flag to enable linking to the shared module
set_target_properties(gdv_core PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
)

# Specify where to look for headers (so that other targets see include/gdv/...)
target_include_directories(gdv_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)