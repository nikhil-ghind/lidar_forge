#include "lidar_gen/config.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

namespace lidar_gen {

GeneratorConfig load_config(const std::string& path) {
    std::ifstream f(path);
    nlohmann::json j;
    f >> j;
    GeneratorConfig cfg;
    cfg.num_scenarios = j.value("num_scenarios", 10000);
    cfg.output_dir = j.value("output_dir", "output");
    cfg.output_format = j.value("output_format", "bin");
    cfg.seed = j.value("seed", (uint64_t)42);
    if (j.contains("lidar")) {
        auto& lj = j["lidar"];
        cfg.lidar.num_beams = lj.value("num_beams", 64);
        cfg.lidar.max_range = lj.value("max_range", 120.0f);
    }
    return cfg;
}

}  // namespace lidar_gen
