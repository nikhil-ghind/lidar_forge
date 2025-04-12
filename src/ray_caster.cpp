#include "lidar_gen/ray_caster.hpp"
#include <cmath>
#include <limits>

namespace lidar_gen {

std::optional<float> ray_aabb_intersection(const Ray& ray, const AABB& aabb) {
    float tmin = -std::numeric_limits<float>::infinity();
    float tmax = std::numeric_limits<float>::infinity();
    for (int i = 0; i < 3; ++i) {
        float inv = 1.0f / ray.direction[i];
        float t1 = (aabb.min_corner[i] - ray.origin[i]) * inv;
        float t2 = (aabb.max_corner[i] - ray.origin[i]) * inv;
        if (t1 > t2) std::swap(t1, t2);
        tmin = std::max(tmin, t1);
        tmax = std::min(tmax, t2);
        if (tmin > tmax) return std::nullopt;
    }
    return tmin > 0 ? std::optional<float>(tmin) : (tmax > 0 ? std::optional<float>(0.0f) : std::nullopt);
}

std::optional<float> ray_plane_intersection(const Ray& ray, const Eigen::Vector4f& plane) {
    float denom = plane.head<3>().dot(ray.direction);
    if (std::abs(denom) < 1e-6f) return std::nullopt;
    float t = -(plane.head<3>().dot(ray.origin) + plane[3]) / denom;
    return t > 0 ? std::optional<float>(t) : std::nullopt;
}

AABB compute_aabb(const SceneObject& obj) {
    Eigen::Vector3f half = obj.dimensions * 0.5f;
    return {obj.position - half, obj.position + half};
}

PointCloud RayCaster::cast(const SceneDescription& scene) {
    PointCloud cloud;
    cloud.frame_id = "lidar";
    cloud.timestamp = scene.timestamp;

    const float PI = 3.14159265f;
    float h_steps = 360.0f / cfg_.horizontal_resolution;

    for (int beam = 0; beam < cfg_.num_beams; ++beam) {
        float vangle = cfg_.vertical_fov_min + (cfg_.vertical_fov_max - cfg_.vertical_fov_min) * beam / (cfg_.num_beams - 1);
        float vrad = vangle * PI / 180.0f;

        for (int h = 0; h < (int)h_steps; ++h) {
            float hrad = h * cfg_.horizontal_resolution * PI / 180.0f;
            Ray ray;
            ray.origin = scene.lidar_position;
            ray.direction = {
                std::cos(vrad) * std::cos(hrad),
                std::cos(vrad) * std::sin(hrad),
                std::sin(vrad)
            };
            ray.direction.normalize();

            float best_t = cfg_.max_range;
            uint8_t best_label = 0;
            bool hit = false;

            // Ground plane
            Eigen::Vector4f ground_plane = {0, 0, 1, 0};
            auto gt = ray_plane_intersection(ray, ground_plane);
            if (gt && *gt < best_t && *gt >= cfg_.min_range) {
                best_t = *gt;
                best_label = 0;
                hit = true;
            }

            // Objects
            for (const auto& obj : scene.objects) {
                auto aabb = compute_aabb(obj);
                auto t = ray_aabb_intersection(ray, aabb);
                if (t && *t < best_t && *t >= cfg_.min_range) {
                    best_t = *t;
                    best_label = obj.label_id;
                    hit = true;
                }
            }

            if (hit) {
                Eigen::Vector3f pt = ray.origin + ray.direction * best_t;
                Point3D p;
                p.x = pt.x(); p.y = pt.y(); p.z = pt.z();
                p.intensity = 0.5f;
                p.label = best_label;
                p.ring_id = (uint16_t)beam;
                cloud.push_back(p);
            }
        }
    }
    return cloud;
}

}  // namespace lidar_gen
