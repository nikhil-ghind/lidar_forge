# LiDAR Forge

C++17 LiDAR point cloud synthesis pipeline with PointNet-based generative model for augmenting autonomous driving datasets.

## Overview

- **Ray-cast engine** — beam-accurate simulation from configurable sensor parameters (64-beam Velodyne HDL-64E profile), AABB intersection, ground-plane intersection
- **Physics-grounded noise** — range-dependent Gaussian noise, random dropout, intensity perturbation
- **Batch generation** — parallel scene generation with JSON scene descriptions, KITTI binary output
- **PointNet GAN** — conditional generator trained on real scans; exports to ONNX
- **Distribution analysis** — point density, range, and intensity histograms comparing real vs generated clouds

## Tech Stack

C++17 · Eigen3 · nlohmann/json · Python 3.11 · PyTorch · ONNX Runtime · CMake · Google Test

## Quickstart

```bash
# Build C++ engine
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Generate dataset
./lidar_generator --config ../config/default_config.json --output ./data --count 1000

# Train PointNet generator
pip install -r requirements.txt
python -m python.training.train --data_dir ./data --epochs 50

# Export to ONNX
python -m python.training.export_model --checkpoint checkpoints/best_generator.pt

# Analyse distributions
python -m python.analysis.distribution_analysis --real_dir ./real_data/velodyne --gen_dir ./data/velodyne
```

## Architecture

```
SceneParser → generate_random()       # random road scene with N objects
    → RayCaster::cast()               # 64×360 beam scan, AABB + ground hits
    → NoiseModel::apply()             # range noise, dropout, intensity jitter
    → generate_annotations()          # per-point labels + 3D bounding boxes
    → write_bin() / write_labels()    # KITTI .bin + JSON
    → PointNetGenerator (PyTorch GAN) # learn real-scan distribution
    → ONNX export                     # deploy in C++
```

## Project Structure

```
lidar_forge/
├── include/lidar_gen/   # Public C++ headers
├── src/                 # C++ implementation + CMakeLists.txt
├── tests/               # Google Test suite
├── python/
│   ├── models/          # PointNetGenerator, PointNetDiscriminator
│   ├── training/        # train.py, export_model.py, dataset.py
│   └── analysis/        # distribution_analysis.py
└── config/              # default_config.json
```
