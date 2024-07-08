cmake_minimum_required(VERSION 3.10)

# Project name and languages
project(ExampleProject VERSION 1.0.0 LANGUAGES CXX ASM)

# Gather all .cpp and .asm files
file(GLOB SOURCES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
    ${PROJECT_SOURCE_DIR}/src/*.S
)

# Set any compile options you need for C++
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# If the assembly files need special flags, set them
set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -fPIC")

set(LIBRARY_NAME "blst")

add_library(blst STATIC ${SOURCES})

# Set the version of the library
set_target_properties(${LIBRARY_NAME} PROPERTIES VERSION ${PROJECT_VERSION})

# Installation rules for the library
install(TARGETS blst
    EXPORT blst_targets
    LIBRARY DESTINATION lib
    INCLUDES DESTINATION include
)

# Define where the headers for the library are
install(FILES ${PROJECT_SOURCE_DIR}/include/blst_t.hpp ${PROJECT_SOURCE_DIR}/include/bytes.h ${PROJECT_SOURCE_DIR}/include/vect.h DESTINATION include/blst)

# Configure the install script to use 'ldconfig' to update the shared library cache
install(CODE "execute_process(COMMAND ldconfig)")
