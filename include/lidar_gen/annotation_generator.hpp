#pragma once
#include "lidar_gen/types.hpp"
#include "lidar_gen/scene.hpp"

namespace lidar_gen {

SceneAnnotation generate_annotations(const PointCloud& cloud, const SceneDescription& scene);

}  // namespace lidar_gen
