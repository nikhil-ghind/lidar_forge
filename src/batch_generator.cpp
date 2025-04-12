#include "lidar_gen/batch_generator.hpp"
#include "lidar_gen/scene.hpp"
#include "lidar_gen/ray_caster.hpp"
#include "lidar_gen/noise_model.hpp"
#include "lidar_gen/output_writer.hpp"
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace lidar_gen {

BatchGenerator::BatchGenerator(const GeneratorConfig& cfg) : cfg_(cfg) {}

BatchResult BatchGenerator::generate_one(int scene_idx) {
    SceneDescription scene = SceneParser::generate_random(cfg_.scene);

    RayCaster caster(cfg_.lidar);
    PointCloud cloud = caster.cast(scene);

    NoiseModel noise(cfg_.noise);
    noise.apply(cloud);

    SceneAnnotation ann = generate_annotations(cloud, scene);

    std::ostringstream oss;
    oss << std::setw(6) << std::setfill('0') << scene_idx;

    return BatchResult{std::move(cloud), std::move(ann), oss.str()};
}

std::vector<BatchResult> BatchGenerator::generate(int count) {
    std::vector<BatchResult> results;
    results.reserve(count);
    for (int i = 0; i < count; ++i)
        results.push_back(generate_one(next_id_++));
    return results;
}

void BatchGenerator::generate_to_disk(int count, const std::string& output_dir,
                                       std::function<void(int, int)> progress_cb) {
    std::filesystem::create_directories(output_dir + "/velodyne");
    std::filesystem::create_directories(output_dir + "/labels");

    for (int i = 0; i < count; ++i) {
        auto result = generate_one(next_id_++);
        write_bin(result.cloud, output_dir + "/velodyne/" + result.scene_id + ".bin");
        write_labels(result.annotation, output_dir + "/labels/" + result.scene_id + ".json");
        if (progress_cb) progress_cb(i + 1, count);
    }
}

}  // namespace lidar_gen
