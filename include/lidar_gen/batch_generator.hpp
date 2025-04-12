#pragma once
#include "lidar_gen/types.hpp"
#include "lidar_gen/config.hpp"
#include <string>
#include <vector>
#include <functional>

namespace lidar_gen {

struct BatchResult {
    PointCloud cloud;
    SceneAnnotation annotation;
    std::string scene_id;
};

class BatchGenerator {
public:
    explicit BatchGenerator(const GeneratorConfig& cfg);

    std::vector<BatchResult> generate(int count);

    void generate_to_disk(int count, const std::string& output_dir,
                          std::function<void(int, int)> progress_cb = nullptr);

private:
    GeneratorConfig cfg_;
    int next_id_{0};

    BatchResult generate_one(int scene_idx);
};

}  // namespace lidar_gen
