#include <gtest/gtest.h>
#include "lidar_gen/noise_model.hpp"
#include "lidar_gen/types.hpp"
#include "lidar_gen/config.hpp"

using namespace lidar_gen;

static PointCloud make_grid_cloud(int n = 100) {
    PointCloud cloud;
    for (int i = 0; i < n; ++i) {
        Point3D p;
        p.x = static_cast<float>(i) * 0.1f;
        p.y = 0.0f;
        p.z = 0.0f;
        p.intensity = 0.5f;
        cloud.points.push_back(p);
    }
    return cloud;
}

TEST(NoiseModelTest, ApplyDoesNotCrash) {
    NoiseConfig cfg;
    cfg.range_sigma        = 0.02f;
    cfg.dropout_rate       = 0.02f;
    cfg.intensity_sigma    = 0.01f;
    cfg.enable_range_noise = true;
    cfg.enable_dropout     = false;

    NoiseModel model(cfg);
    PointCloud cloud = make_grid_cloud();
    EXPECT_NO_THROW(model.apply(cloud));
}

TEST(NoiseModelTest, DropoutReducesPoints) {
    NoiseConfig cfg;
    cfg.range_sigma        = 0.0f;
    cfg.dropout_rate       = 0.5f;
    cfg.intensity_sigma    = 0.0f;
    cfg.enable_range_noise = false;
    cfg.enable_dropout     = true;

    NoiseModel model(cfg);
    PointCloud cloud = make_grid_cloud(1000);
    size_t before = cloud.points.size();
    model.apply(cloud);
    EXPECT_LT(cloud.points.size(), before);
}

TEST(NoiseModelTest, IntensityClamped) {
    NoiseConfig cfg;
    cfg.range_sigma        = 0.0f;
    cfg.dropout_rate       = 0.0f;
    cfg.intensity_sigma    = 1.0f;
    cfg.enable_range_noise = false;
    cfg.enable_dropout     = false;

    NoiseModel model(cfg);
    PointCloud cloud = make_grid_cloud(200);
    model.apply(cloud);
    for (const auto& p : cloud.points) {
        EXPECT_GE(p.intensity, 0.0f);
        EXPECT_LE(p.intensity, 1.0f);
    }
}
