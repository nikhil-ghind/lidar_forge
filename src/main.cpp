#include "lidar_gen/config.hpp"
#include "lidar_gen/batch_generator.hpp"
#include <iostream>
#include <string>
#include <stdexcept>

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog << " --config <path> --output <dir> --count <N>\n";
}

int main(int argc, char** argv) {
    std::string config_path = "config/default_config.json";
    std::string output_dir  = "output";
    int count = 100;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_dir = argv[++i];
        else if (arg == "--count" && i + 1 < argc) count = std::stoi(argv[++i]);
        else if (arg == "--help") { usage(argv[0]); return 0; }
    }

    try {
        lidar_gen::GeneratorConfig cfg = lidar_gen::load_config(config_path);
        lidar_gen::BatchGenerator gen(cfg);

        std::cout << "Generating " << count << " scenes to " << output_dir << " ...\n";
        gen.generate_to_disk(count, output_dir,
            [](int done, int total) {
                if (done % 10 == 0 || done == total)
                    std::cout << "\r  " << done << "/" << total << std::flush;
            });
        std::cout << "\nDone.\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
