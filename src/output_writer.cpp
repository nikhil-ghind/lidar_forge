#include "lidar_gen/types.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

namespace lidar_gen {

void write_bin(const PointCloud& cloud, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    for (const auto& p : cloud.points) {
        f.write(reinterpret_cast<const char*>(&p.x), 4);
        f.write(reinterpret_cast<const char*>(&p.y), 4);
        f.write(reinterpret_cast<const char*>(&p.z), 4);
        f.write(reinterpret_cast<const char*>(&p.intensity), 4);
    }
}

void write_labels(const SceneAnnotation& ann, const std::string& path) {
    nlohmann::json j;
    j["point_labels"] = ann.point_labels;
    nlohmann::json boxes = nlohmann::json::array();
    for (const auto& b : ann.boxes) {
        boxes.push_back({{"class", b.class_name}, {"cx", b.cx}, {"cy", b.cy}, {"cz", b.cz},
                         {"l", b.length}, {"w", b.width}, {"h", b.height}, {"yaw", b.yaw}});
    }
    j["bounding_boxes"] = boxes;
    std::ofstream f(path);
    f << j.dump(2);
}

}  // namespace lidar_gen
