#include "lidar_gen/types.hpp"
#include "lidar_gen/scene.hpp"
#include <cmath>

namespace lidar_gen {

SceneAnnotation generate_annotations(const PointCloud& cloud, const SceneDescription& scene) {
    SceneAnnotation ann;
    for (const auto& p : cloud.points) {
        ann.point_labels.push_back(p.label);
    }
    for (const auto& obj : scene.objects) {
        BoundingBox3D box;
        box.cx = obj.position.x(); box.cy = obj.position.y(); box.cz = obj.position.z();
        box.length = obj.dimensions.x(); box.width = obj.dimensions.y(); box.height = obj.dimensions.z();
        box.yaw = obj.yaw;
        box.label = obj.label_id;
        box.class_name = obj.class_name;
        // Count points inside box
        int count = 0;
        for (const auto& p : cloud.points) {
            float dx = p.x - box.cx, dy = p.y - box.cy;
            if (std::abs(dx) < box.length / 2 && std::abs(dy) < box.width / 2) count++;
        }
        (void)count;
        ann.boxes.push_back(box);
    }
    return ann;
}

}  // namespace lidar_gen
