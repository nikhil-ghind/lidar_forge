#pragma once
#include "types.hpp"
#include "config.hpp"
#include "scene.hpp"
#include <Eigen/Core>
#include <optional>

namespace lidar_gen {

struct Ray {
    Eigen::Vector3f origin;
    Eigen::Vector3f direction;
};

struct AABB {
    Eigen::Vector3f min_corner;
    Eigen::Vector3f max_corner;
};

std::optional<float> ray_aabb_intersection(const Ray& ray, const AABB& aabb);
std::optional<float> ray_plane_intersection(const Ray& ray, const Eigen::Vector4f& plane);
AABB compute_aabb(const SceneObject& obj);

class RayCaster {
public:
    explicit RayCaster(const LidarConfig& cfg) : cfg_(cfg) {}
    PointCloud cast(const SceneDescription& scene);

private:
    LidarConfig cfg_;
};

}  // namespace lidar_gen
