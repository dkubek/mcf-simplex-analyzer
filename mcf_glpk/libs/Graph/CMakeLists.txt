cmake_minimum_required(VERSION 3.0.0)

# Define variables for library
set(PROJECT_NAME Graph)
set(LIBRARY_NAME Graph)
set(LIBRARY_HEADERS_DIR include/${LIBRARY_NAME})
set(LIBRARY_HEADERS ${LIBRARY_HEADERS_DIR}/Graph.h)
set(LIBRARY_SOURCE Graph.C)

project(${PROJECT_NAME})

# Make a new STATIC library
#   STATIC -> will be embeded in binary
add_library(${LIBRARY_NAME} STATIC
    ${LIBRARY_HEADERS}
    ${LIBRARY_SOURCE}
)

# Expose Public Header files, we would not be able to include library header
# files otherwise
target_include_directories(${LIBRARY_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${LIBRARY_NAME}>
    $<INSTALL_INTERFACE:include/${LIBRARY_NAME}>
)
