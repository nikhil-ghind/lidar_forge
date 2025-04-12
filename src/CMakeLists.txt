set(LIB_SOURCES
    config.cpp
    scene.cpp
    ray_caster.cpp
    noise_model.cpp
    annotation_generator.cpp
    output_writer.cpp
    batch_generator.cpp
)

add_library(lidar_gen_lib STATIC ${LIB_SOURCES})
target_include_directories(lidar_gen_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(lidar_gen_lib PUBLIC Eigen3::Eigen nlohmann_json::nlohmann_json)

add_executable(lidar_generator main.cpp)
target_link_libraries(lidar_generator PRIVATE lidar_gen_lib)
