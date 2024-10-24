cmake_minimum_required(VERSION 3.10)
project(LERW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(lerw_cpp main.cpp)
target_compile_options(lerw_cpp PRIVATE -Wall -Wextra)
