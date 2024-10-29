cmake_minimum_required(VERSION 3.10)
project(LERW_Simulation CXX)

# Enable C++17 features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set optimization flags based on compiler
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto -ffast-math")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -fomit-frame-pointer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /GL /fp:fast /arch:AVX2")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
endif()

# Add include directory
include_directories(${PROJECT_SOURCE_DIR}/include)

# Collect source files
set(SOURCES
    src/lerw_estimation.cpp
    src/lerw_3d.cpp
    src/random_utils.cpp
)

# Only add 2D simulation if needed
if(NOT DEFINED DISABLE_2D)
    list(APPEND SOURCES src/lerw_2d.cpp)
endif()

# Add the executable
add_executable(lerw_simulation ${SOURCES})

# Find and link OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(lerw_simulation PRIVATE OpenMP::OpenMP_CXX)
