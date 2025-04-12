find_package(GTest REQUIRED)

add_executable(test_lidar_forge
    test_ray_caster.cpp
    test_noise_model.cpp
    test_annotation.cpp
)

target_include_directories(test_lidar_forge PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_lidar_forge PRIVATE lidar_gen_lib GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_lidar_forge)
