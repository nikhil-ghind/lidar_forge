#pragma once
#include <string>
#include <vector>
#include <Eigen/Core>

namespace lidar_gen {

struct SceneObject {
    std::string class_name;
    uint8_t label_id;
    Eigen::Vector3f position;
    Eigen::Vector3f dimensions;
    float yaw;
    std::string mesh_path;
};

struct SceneGenConfig {
    int min_objects = 2;
    int max_objects = 15;
    float placement_range = 50.0f;
};

struct SceneDescription {
    std::vector<SceneObject> objects;
    Eigen::Vector3f lidar_position = {0, 0, 1.8f};
    float lidar_height = 1.8f;
    std::string ground_plane = "flat";
    double timestamp = 0.0;
};

class SceneParser {
public:
    SceneDescription parse(const std::string& json_path);
    SceneDescription generate_random(const SceneGenConfig& config);
};

}  // namespace lidar_gen
