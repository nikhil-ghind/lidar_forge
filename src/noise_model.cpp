#include "lidar_gen/noise_model.hpp"
#include <cmath>

namespace lidar_gen {

PointCloud NoiseModel::apply(const PointCloud& cloud) {
    PointCloud result;
    result.frame_id = cloud.frame_id;
    result.timestamp = cloud.timestamp;

    std::normal_distribution<float> range_noise(0.0f, cfg_.range_noise_stddev);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::normal_distribution<float> intensity_noise(0.0f, cfg_.intensity_noise_stddev);

    for (const auto& p : cloud.points) {
        if (uniform(rng_) < cfg_.dropout_rate) continue;

        float range = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        float scale = 1.0f;
        if (range > 0) {
            float noise = range_noise(rng_) * (1.0f + range / 120.0f);
            scale = (range + noise) / range;
        }

        Point3D noisy = p;
        noisy.x *= scale; noisy.y *= scale; noisy.z *= scale;
        noisy.intensity = std::max(0.0f, std::min(1.0f, p.intensity + intensity_noise(rng_)));
        result.push_back(noisy);
    }
    return result;
}

}  // namespace lidar_gen
