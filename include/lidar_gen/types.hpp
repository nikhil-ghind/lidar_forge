#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace lidar_gen {

struct Point3D {
    float x, y, z;
    float intensity;
    uint8_t label;
    uint16_t ring_id;
};

struct PointCloud {
    std::vector<Point3D> points;
    std::string frame_id;
    double timestamp = 0.0;
    void push_back(const Point3D& p) { points.push_back(p); }
    size_t size() const { return points.size(); }
};

struct BoundingBox3D {
    float cx, cy, cz;
    float length, width, height;
    float yaw;
    uint8_t label;
    std::string class_name;
};

struct SceneAnnotation {
    std::vector<BoundingBox3D> boxes;
    std::vector<uint8_t> point_labels;
};

}  // namespace lidar_gen
