# LiDAR Forge

## Project Overview

A C++ pipeline that adapts a PointNet-based deep learning model to synthesize realistic LiDAR point clouds for simulation testing of autonomous vehicle perception systems. Supports configurable noise models, occlusion simulation, and intensity falloff based on physical LiDAR characteristics. Generates 10,000+ labeled scenarios with ground-truth annotations for downstream training and validation.

**Key Goals:**
- Generate physically plausible synthetic LiDAR point clouds from 3D scene descriptions
- Adapt PointNet architecture for point cloud generation (decoder/generator variant)
- Configurable noise, occlusion, and intensity falloff models based on real LiDAR specifications
- Produce 10k+ labeled scenarios with semantic labels and bounding boxes
- Statistical validation ensuring generated distributions match real-world LiDAR data
- Efficient C++ pipeline for batch generation with Python bindings for ML integration

## Tech Stack

- C++17 (GCC 12+ or Clang 15+)
- CMake 3.22+
- PyTorch C++ (libtorch) for PointNet inference in C++
- Python 3.11+ (model training, statistical analysis)
- PyTorch 2.1+ (Python, for PointNet training)
- PCL (Point Cloud Library) 1.13+ for point cloud I/O and visualization
- Eigen 3.4 for linear algebra
- nlohmann/json for config parsing
- NumPy, SciPy, Matplotlib (statistical distribution analysis)
- Open3D (optional, for visualization)

## Architecture Overview

```
[Scene Description (JSON/YAML)]
       |
       v
[Scene Parser] --> [3D Object Placement]
       |
       v
[Ray Casting Engine] --> [Raw Point Cloud]
       |
       v
[PointNet Generator] --> [Enhanced/Synthesized Points]
       |
       v
[Noise Model] --> [Occlusion Model] --> [Intensity Model]
       |
       v
[Annotation Generator] --> [Labeled Point Cloud + Metadata]
       |
       v
[Output Writer] --> PCD/PLY/BIN files + label files
```

**Components:**
1. **Scene Parser** - Reads scene descriptions, places 3D objects in the world.
2. **Ray Casting Engine** - Simulates LiDAR beam scanning pattern (Velodyne-style).
3. **PointNet Generator** - Neural network that refines/synthesizes realistic point distributions.
4. **Noise Model** - Adds range-dependent Gaussian noise, dropout, multi-path reflections.
5. **Occlusion Model** - Computes visibility, removes occluded points.
6. **Intensity Model** - Computes return intensity based on range, angle, material.
7. **Annotation Generator** - Produces per-point semantic labels, 3D bounding boxes.

---

## Phase 1: Project Setup and Core Data Structures

**Goal:** Set up the C++ build system and define core data structures for point clouds and scenes.

### Tasks

1. Create `CMakeLists.txt` (root):
   - cmake_minimum_required(VERSION 3.22), project(synthetic_lidar_generator).
   - C++17 standard, warnings enabled.
   - Find packages: Eigen3, PCL, Torch (libtorch), nlohmann_json.
   - Subdirectories: src, tests, python.

2. Create `src/CMakeLists.txt`:
   - Add library `lidar_gen_lib` from source files.
   - Link: Eigen3::Eigen, ${PCL_LIBRARIES}, ${TORCH_LIBRARIES}, nlohmann_json::nlohmann_json.
   - Add executable `lidar_generator` from main.cpp, link lidar_gen_lib.

3. Create `include/lidar_gen/types.hpp`:
   - `struct Point3D { float x, y, z; float intensity; uint8_t label; uint16_t ring_id; }`.
   - `struct PointCloud { std::vector<Point3D> points; std::string frame_id; double timestamp; }`.
   - `struct BoundingBox3D { float cx, cy, cz; float length, width, height; float yaw; uint8_t label; std::string class_name; }`.
   - `struct SceneAnnotation { std::vector<BoundingBox3D> boxes; std::vector<uint8_t> point_labels; }`.

4. Create `include/lidar_gen/scene.hpp` and `src/scene.cpp`:
   - `struct SceneObject { std::string class_name; uint8_t label_id; Eigen::Vector3f position; Eigen::Vector3f dimensions; float yaw; std::string mesh_path; }`.
   - `struct SceneDescription { std::vector<SceneObject> objects; Eigen::Vector3f lidar_position; float lidar_height; std::string ground_plane; // flat, sloped, uneven; double timestamp; }`.
   - Class `SceneParser`:
     - Method: `SceneDescription parse(const std::string& json_path)` - loads scene from JSON file.
     - Method: `SceneDescription generate_random(const SceneGenConfig& config)` - generates random scene with configurable object counts, placement ranges.

5. Create `include/lidar_gen/config.hpp` and `src/config.cpp`:
   - `struct LidarConfig`: num_beams (e.g., 64), vertical_fov_min (-25 deg), vertical_fov_max (15 deg), horizontal_resolution (0.2 deg), max_range (120m), min_range (0.5m), rotation_rate (10 Hz).
   - `struct NoiseConfig`: range_noise_stddev (0.02m), dropout_rate (0.01), intensity_noise_stddev (0.05).
   - `struct GeneratorConfig`: num_scenarios (10000), output_dir, output_format (pcd/ply/bin), lidar_config, noise_config, seed.
   - Function: `GeneratorConfig load_config(const std::string& path)` - loads from JSON.

6. Create `config/default_config.json`:
   - Default configuration with Velodyne VLP-64 specifications.

7. Create `tests/test_types.cpp`:
   - Test PointCloud construction, push_back, size.
   - Test SceneParser loads valid JSON.

8. Build: `mkdir build && cd build && cmake .. && make -j$(nproc)`.

---

## Phase 2: Ray Casting Engine

**Goal:** Implement LiDAR beam simulation via ray casting against scene geometry.

### Tasks

1. Create `include/lidar_gen/ray_caster.hpp` and `src/ray_caster.cpp`:
   - Class `RayCaster`:
     - Constructor: takes LidarConfig.
     - Method: `PointCloud cast(const SceneDescription& scene) -> PointCloud`:
       - For each beam (ring): compute vertical angle from fov range.
       - For each horizontal step: compute azimuth angle.
       - Cast ray from lidar_position in direction (azimuth, elevation).
       - Test intersection with ground plane and each scene object's bounding box (AABB).
       - For hit: compute range, create Point3D with (x,y,z), assign label from hit object, compute initial intensity.
       - For miss (no hit within max_range): optionally add no-return.

2. Create `include/lidar_gen/geometry.hpp` and `src/geometry.cpp`:
   - `struct Ray { Eigen::Vector3f origin; Eigen::Vector3f direction; }`.
   - `struct AABB { Eigen::Vector3f min_corner; Eigen::Vector3f max_corner; }`.
   - Function: `std::optional<float> ray_aabb_intersection(const Ray& ray, const AABB& aabb)` - returns distance to intersection or nullopt.
   - Function: `std::optional<float> ray_plane_intersection(const Ray& ray, const Eigen::Vector4f& plane)`.
   - Function: `AABB compute_aabb(const SceneObject& obj)` - computes axis-aligned bounding box from object position, dimensions, yaw.

3. Create `tests/test_ray_caster.cpp`:
   - Test ray-AABB intersection with known geometry.
   - Test ray-plane intersection (ground plane hit).
   - Test full cast with single object: correct number of points, all within object bounds.
   - Test cast with empty scene: only ground plane hits.

---

## Phase 3: PointNet Generator Model

**Goal:** Train a PointNet variant in Python, export to TorchScript, and integrate into C++ pipeline.

### Tasks

1. Create `python/models/pointnet_generator.py`:
   - Class `PointNetGenerator(nn.Module)`:
     - Encoder: shared MLP (3 -> 64 -> 128 -> 256 -> 512 -> 1024) with batch norm and ReLU.
     - Global feature: max pooling over points.
     - Decoder: MLP (1024 + class_embedding -> 512 -> 256 -> 128 -> N*3) outputting point coordinates.
     - Forward: takes (coarse_points, class_label) -> refined_points.
   - The generator takes a coarse ray-cast point cloud and produces a refined, realistic version.

2. Create `python/models/pointnet_discriminator.py`:
   - Class `PointNetDiscriminator(nn.Module)`:
     - PointNet encoder -> global feature -> MLP classifier (real/fake).
     - Used during training for adversarial loss (optional, for quality).

3. Create `python/training/dataset.py`:
   - Class `LidarDataset(torch.utils.data.Dataset)`:
     - Loads real LiDAR point clouds (e.g., from KITTI or nuScenes format).
     - Normalizes, subsamples to fixed N points.
     - Returns (points, label).

4. Create `python/training/train.py`:
   - Training loop:
     - Load real LiDAR data as ground truth.
     - Generate coarse point clouds via simplified ray casting.
     - Train PointNetGenerator to refine coarse -> realistic using Chamfer distance loss + Earth Mover's distance.
     - Optimizer: Adam, lr=1e-4, batch_size=32, epochs=100.
     - Save best model checkpoint.
   - Command: `python python/training/train.py --data_dir data/real_lidar --epochs 100 --output models/pointnet_gen.pt`.

5. Create `python/training/export_model.py`:
   - Load trained model checkpoint.
   - Export to TorchScript: `torch.jit.script(model)` or `torch.jit.trace(model, example_input)`.
   - Save: `models/pointnet_gen_scripted.pt`.
   - Command: `python python/training/export_model.py --checkpoint models/pointnet_gen.pt --output models/pointnet_gen_scripted.pt`.

6. Create `include/lidar_gen/pointnet_refiner.hpp` and `src/pointnet_refiner.cpp`:
   - Class `PointNetRefiner`:
     - Constructor: `PointNetRefiner(const std::string& model_path)` - loads TorchScript model via `torch::jit::load()`.
     - Method: `PointCloud refine(const PointCloud& coarse, uint8_t object_label) -> PointCloud`:
       - Convert point cloud to torch::Tensor (N x 3).
       - Run inference: `auto output = model_.forward({input_tensor, label_tensor})`.
       - Convert output tensor back to PointCloud.
       - Preserve original metadata (labels, ring_id).

7. Create `tests/test_pointnet_refiner.cpp`:
   - Test model loads without error.
   - Test refine produces output with same number of points.
   - Test output points are within reasonable range.

---

## Phase 4: Noise, Occlusion, and Intensity Models

**Goal:** Add physically-based noise, occlusion simulation, and intensity falloff.

### Tasks

1. Create `include/lidar_gen/noise_model.hpp` and `src/noise_model.cpp`:
   - Class `NoiseModel`:
     - Constructor: takes NoiseConfig.
     - Method: `PointCloud apply(const PointCloud& cloud) -> PointCloud`:
       - Range noise: add Gaussian noise to each point's range, N(0, stddev). Stddev scales with range: `noise = base_stddev * (1 + range/max_range)`.
       - Dropout: randomly remove points with probability dropout_rate.
       - Multi-path noise: with low probability (0.1%), add ghost points at 2x range.
       - Quantization: round range to LiDAR's range resolution (e.g., 0.002m).

2. Create `include/lidar_gen/occlusion_model.hpp` and `src/occlusion_model.cpp`:
   - Class `OcclusionModel`:
     - Method: `PointCloud apply(const PointCloud& cloud, const SceneDescription& scene) -> PointCloud`:
       - For each point, check if the ray from LiDAR to that point is blocked by a closer object.
       - Use depth buffer approach: discretize azimuth/elevation into grid, keep only closest point per cell.
       - Self-occlusion: points on far side of objects are removed.
       - Partial occlusion: at object edges, randomly thin points to simulate beam divergence.

3. Create `include/lidar_gen/intensity_model.hpp` and `src/intensity_model.cpp`:
   - Class `IntensityModel`:
     - Method: `void apply(PointCloud& cloud, const SceneDescription& scene)`:
       - Base intensity from material reflectivity: road=0.3, vehicle=0.8, vegetation=0.5, pedestrian=0.4.
       - Range falloff: intensity *= 1.0 / (range * range) (inverse square law), clamped to [0, 1].
       - Incidence angle effect: intensity *= cos(incidence_angle).
       - Add Gaussian noise to intensity values.

4. Create `include/lidar_gen/post_processor.hpp` and `src/post_processor.cpp`:
   - Class `PostProcessor`:
     - Chains: PointNetRefiner -> NoiseModel -> OcclusionModel -> IntensityModel.
     - Method: `PointCloud process(const PointCloud& raw, const SceneDescription& scene) -> PointCloud`.

5. Create `tests/test_noise_model.cpp`:
   - Test dropout reduces point count by approximately dropout_rate.
   - Test range noise adds perturbation (mean ~0, std matches config).
   - Test no points outside max_range after noise.

6. Create `tests/test_occlusion_model.cpp`:
   - Test: object behind another object has fewer points.
   - Test: visible object retains all points.

---

## Phase 5: Annotation Generation and Output

**Goal:** Generate ground-truth labels and write output files in standard formats.

### Tasks

1. Create `include/lidar_gen/annotation_generator.hpp` and `src/annotation_generator.cpp`:
   - Class `AnnotationGenerator`:
     - Method: `SceneAnnotation generate(const PointCloud& cloud, const SceneDescription& scene) -> SceneAnnotation`:
       - Per-point semantic labels already assigned during ray casting (from hit object).
       - Generate 3D bounding boxes from SceneObject positions/dimensions.
       - Compute per-box point count (number of points inside each box).
       - Flag boxes with < 5 points as "heavily occluded".

2. Create `include/lidar_gen/output_writer.hpp` and `src/output_writer.cpp`:
   - Class `OutputWriter`:
     - Method: `void write_pcd(const PointCloud& cloud, const std::string& path)` - PCL PCD format.
     - Method: `void write_ply(const PointCloud& cloud, const std::string& path)` - PLY format.
     - Method: `void write_bin(const PointCloud& cloud, const std::string& path)` - KITTI binary format (x,y,z,intensity as float32).
     - Method: `void write_labels(const SceneAnnotation& ann, const std::string& path)` - JSON with bounding boxes and per-point labels.
     - Method: `void write_metadata(const SceneDescription& scene, const std::string& path)` - scene metadata JSON.

3. Create `include/lidar_gen/batch_generator.hpp` and `src/batch_generator.cpp`:
   - Class `BatchGenerator`:
     - Constructor: takes GeneratorConfig.
     - Method: `void generate(int num_scenarios)`:
       - For each scenario i = 0..num_scenarios:
         - Generate or load scene description.
         - Run ray casting.
         - Run post-processing (PointNet + noise + occlusion + intensity).
         - Generate annotations.
         - Write output files: `output_dir/scenario_{i:06d}.{pcd,bin}`, `output_dir/scenario_{i:06d}_labels.json`, `output_dir/scenario_{i:06d}_meta.json`.
       - Print progress every 100 scenarios.
       - Support multi-threaded generation (std::async or thread pool) for parallelism.

4. Create `src/main.cpp`:
   - Parse command-line arguments: --config, --num-scenarios, --output-dir, --format, --threads.
   - Load config.
   - Instantiate BatchGenerator.
   - Run generation.
   - Print summary: total scenarios, total points, elapsed time.

5. Create `tests/test_output_writer.cpp`:
   - Test write_pcd creates valid PCD file that PCL can load.
   - Test write_bin creates correct binary format.
   - Test write_labels creates valid JSON.

---

## Phase 6: Statistical Validation and Analysis

**Goal:** Validate that generated point clouds statistically match real-world LiDAR data distributions.

### Tasks

1. Create `python/analysis/distribution_analysis.py`:
   - Function: `analyze_point_cloud(cloud_path) -> dict`:
     - Compute: total point count, points per ring, range histogram, intensity histogram, point density vs range, angular distribution.
   - Function: `compare_distributions(generated_stats, real_stats) -> dict`:
     - KS test (Kolmogorov-Smirnov) on range distributions.
     - Chi-squared test on intensity histograms.
     - Jensen-Shannon divergence on point density distributions.
     - Return p-values and divergence scores.

2. Create `python/analysis/visualize_stats.py`:
   - Function: `plot_range_distribution(generated, real, output_path)` - overlay histograms.
   - Function: `plot_intensity_distribution(generated, real, output_path)`.
   - Function: `plot_point_density_heatmap(cloud, output_path)` - bird's-eye view density.
   - Function: `plot_beam_pattern(cloud, output_path)` - ring-level visualization.

3. Create `python/analysis/batch_validate.py`:
   - Script that:
     - Loads N generated and N real point clouds.
     - Runs distribution analysis on each.
     - Aggregates statistics.
     - Runs comparison tests.
     - Outputs validation report with pass/fail criteria.
   - Command: `python python/analysis/batch_validate.py --generated output/ --real data/real_lidar/ --num-samples 100`.

4. Create `python/analysis/scene_statistics.py`:
   - Analyze generated scenario diversity:
     - Object count distribution.
     - Object class balance.
     - Spatial placement coverage.
     - Occlusion rate distribution.
   - Ensure scenarios span the desired range of complexity.

5. Create `tests/test_distribution_analysis.py` (Python pytest):
   - Test KS test returns low p-value for clearly different distributions.
   - Test KS test returns high p-value for same distribution with noise.
