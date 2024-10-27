cmake_minimum_required(VERSION 3.10)
project(LERW_Simulation VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add OpenMP support
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Add include directory
include_directories(${PROJECT_SOURCE_DIR}/include)

# Main executable
add_executable(lerw_sim 
    src/random_utils.cpp    # Make sure this line is here
    src/lerw_2d.cpp
    src/lerw_3d.cpp
    src/lerw_estimation.cpp
)

# Add include directories specifically for the executable
target_include_directories(lerw_sim PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Testing setup
enable_testing()
add_subdirectory(tests)