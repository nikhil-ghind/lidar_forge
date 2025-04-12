#pragma once
#include "lidar_gen/types.hpp"
#include <string>

namespace lidar_gen {

void write_bin(const PointCloud& cloud, const std::string& path);
void write_labels(const SceneAnnotation& ann, const std::string& path);

}  // namespace lidar_gen
