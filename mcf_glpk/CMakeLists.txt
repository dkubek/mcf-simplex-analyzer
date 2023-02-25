cmake_minimum_required(VERSION 3.11)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

# Setup the project
project(mcfglpk
    VERSION 0.1
    DESCRIPTION "Solve MCF problems using GLPK"
    LANGUAGES CXX)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add external libraries
add_subdirectory(libs/Graph)

# Add custom GLPK library
set(GLPK_WITH_GMP ON)
add_subdirectory(libs/glpk)

# Add cxxopts
add_subdirectory(libs/cxxopts)

# $source_files is a space-delimited list of filenames.
add_executable(mcfglpk src/mcfglpk.cpp)
add_executable(convert src/convert.cpp)

target_compile_features(mcfglpk PUBLIC cxx_std_17)

# IMPORTANT: Make sure to link library to executable *after* it has been added
target_link_libraries(mcfglpk PRIVATE Graph)
target_link_libraries(mcfglpk PRIVATE gmp)
target_link_libraries(mcfglpk PRIVATE gmpxx)
target_link_libraries(mcfglpk PRIVATE glpk-debug)
target_include_directories(mcfglpk PUBLIC cxxopts)
target_link_libraries(mcfglpk PUBLIC cxxopts)

target_link_libraries(convert PRIVATE Graph)
