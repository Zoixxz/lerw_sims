cmake_minimum_required(VERSION 3.16)
project(LongRange2D)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# # Find required packages
find_package(TBB REQUIRED)
find_package(Threads REQUIRED)

# Include directories for external libraries
# add_subdirectory(abseil-cpp)

# Add the executable
add_executable(main main.cpp)

add_compile_options(-Wall -Wextra -O3 -march=native -mtune=native -flto)

# Link required libraries
target_link_libraries(main TBB::tbb Threads::Threads)

# cmake_minimum_required(VERSION 3.16)

# project(my_project)

# # Abseil requires C++14
# set(CMAKE_CXX_STANDARD 17)

# # Process Abseil's CMake build system
# add_subdirectory(abseil-cpp)

# add_executable(test test.cpp)

# # Declare dependency on the absl::strings library
# target_link_libraries(test absl::strings)