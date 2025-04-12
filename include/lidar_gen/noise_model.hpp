#pragma once
#include "types.hpp"
#include "config.hpp"
#include <random>

namespace lidar_gen {

class NoiseModel {
public:
    explicit NoiseModel(const NoiseConfig& cfg, uint64_t seed = 42)
        : cfg_(cfg), rng_(seed) {}
    PointCloud apply(const PointCloud& cloud);

private:
    NoiseConfig cfg_;
    std::mt19937_64 rng_;
};

}  // namespace lidar_gen
