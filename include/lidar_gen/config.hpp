#pragma once
#include <string>
#include <cstdint>

namespace lidar_gen {

struct LidarConfig {
    int num_beams = 64;
    float vertical_fov_min = -25.0f;   // degrees
    float vertical_fov_max = 15.0f;    // degrees
    float horizontal_resolution = 0.2f; // degrees
    float max_range = 120.0f;
    float min_range = 0.5f;
    float rotation_rate = 10.0f;        // Hz
};

struct NoiseConfig {
    float range_noise_stddev = 0.02f;
    float dropout_rate = 0.01f;
    float intensity_noise_stddev = 0.05f;
};

struct GeneratorConfig {
    int num_scenarios = 10000;
    std::string output_dir = "output";
    std::string output_format = "bin"; // pcd|ply|bin
    LidarConfig lidar;
    NoiseConfig noise;
    uint64_t seed = 42;
};

GeneratorConfig load_config(const std::string& path);

}  // namespace lidar_gen
