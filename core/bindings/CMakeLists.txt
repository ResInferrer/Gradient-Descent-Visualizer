cmake_minimum_required(VERSION 3.15)
project(gdv_bindings)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Create a Python module from our wrapper file
pybind11_add_module(gdv_vis MODULE pybind_wrapper.cpp)

# Link our C++ library
target_link_libraries(gdv_vis PRIVATE gdv_core)

# Where to put the compiled module
set_target_properties(gdv_vis PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Copy the module to the project folder (for easy import)
add_custom_command(TARGET gdv_vis POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:gdv_vis>
        ${CMAKE_SOURCE_DIR}/gdv_vis.so
    COMMENT "Copying module to project root"
)