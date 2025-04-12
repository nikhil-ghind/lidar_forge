#include "lidar_gen/scene.hpp"
#include <fstream>
#include <random>
#include <nlohmann/json.hpp>

namespace lidar_gen {

SceneDescription SceneParser::parse(const std::string& json_path) {
    std::ifstream f(json_path);
    nlohmann::json j;
    f >> j;
    SceneDescription scene;
    scene.timestamp = j.value("timestamp", 0.0);
    scene.ground_plane = j.value("ground_plane", "flat");
    for (auto& obj_j : j.value("objects", nlohmann::json::array())) {
        SceneObject obj;
        obj.class_name = obj_j.value("class_name", "vehicle");
        obj.label_id = obj_j.value("label_id", 1);
        auto pos = obj_j.value("position", std::vector<float>{0, 0, 0});
        obj.position = {pos[0], pos[1], pos[2]};
        auto dim = obj_j.value("dimensions", std::vector<float>{4, 2, 1.5f});
        obj.dimensions = {dim[0], dim[1], dim[2]};
        obj.yaw = obj_j.value("yaw", 0.0f);
        scene.objects.push_back(obj);
    }
    return scene;
}

SceneDescription SceneParser::generate_random(const SceneGenConfig& cfg) {
    static std::mt19937 rng(42);
    std::uniform_int_distribution<int> n_dist(cfg.min_objects, cfg.max_objects);
    std::uniform_real_distribution<float> pos_dist(-cfg.placement_range, cfg.placement_range);
    std::uniform_real_distribution<float> yaw_dist(0, 6.28f);

    SceneDescription scene;
    scene.lidar_position = {0, 0, 1.8f};
    int n = n_dist(rng);
    static const char* classes[] = {"vehicle", "pedestrian", "cyclist", "vegetation"};
    static uint8_t labels[] = {1, 2, 3, 4};
    std::uniform_int_distribution<int> cls_dist(0, 3);

    for (int i = 0; i < n; ++i) {
        SceneObject obj;
        int ci = cls_dist(rng);
        obj.class_name = classes[ci];
        obj.label_id = labels[ci];
        obj.position = {pos_dist(rng), pos_dist(rng), 0.75f};
        obj.dimensions = {4.0f, 2.0f, 1.5f};
        obj.yaw = yaw_dist(rng);
        scene.objects.push_back(obj);
    }
    return scene;
}

}  // namespace lidar_gen
