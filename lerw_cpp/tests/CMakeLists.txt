find_package(GTest REQUIRED)
include_directories(
    ${GTEST_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(test_lerw 
    test_lerw.cpp
    ${PROJECT_SOURCE_DIR}/src/random_utils.cpp
    ${PROJECT_SOURCE_DIR}/src/lerw_2d.cpp
    ${PROJECT_SOURCE_DIR}/src/lerw_3d.cpp
)

target_link_libraries(test_lerw ${GTEST_LIBRARIES} pthread)
add_test(NAME test_lerw COMMAND test_lerw)