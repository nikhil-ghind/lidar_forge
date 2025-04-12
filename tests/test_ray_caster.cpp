#include <gtest/gtest.h>
#include "lidar_gen/ray_caster.hpp"
#include "lidar_gen/scene.hpp"
#include "lidar_gen/config.hpp"

using namespace lidar_gen;

static LidarConfig default_lidar_cfg() {
    LidarConfig c;
    c.num_beams        = 16;
    c.horizontal_res   = 1.0f;
    c.vertical_fov_min = -15.0f;
    c.vertical_fov_max = 15.0f;
    c.max_range        = 50.0f;
    c.min_range        = 0.5f;
    c.sensor_height    = 1.8f;
    return c;
}

TEST(RayCasterTest, EmptySceneProducesGroundPoints) {
    SceneDescription scene;
    scene.ground_z = -1.8f;

    LidarConfig cfg = default_lidar_cfg();
    RayCaster caster(cfg);
    PointCloud cloud = caster.cast(scene);

    EXPECT_GT(cloud.points.size(), 0u);
}

TEST(RayCasterTest, AllPointsWithinMaxRange) {
    SceneDescription scene;
    scene.ground_z = -1.8f;
    LidarConfig cfg = default_lidar_cfg();
    RayCaster caster(cfg);
    PointCloud cloud = caster.cast(scene);

    for (const auto& p : cloud.points) {
        float r = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        EXPECT_LE(r, cfg.max_range + 1e-3f);
    }
}

TEST(RayCasterTest, BoxOccludesPoints) {
    SceneDescription scene;
    scene.ground_z = -1.8f;
    SceneObject box;
    box.class_name = "Car";
    box.cx = 5.0f; box.cy = 0.0f; box.cz = 0.0f;
    box.length = 4.0f; box.width = 2.0f; box.height = 1.5f;
    box.yaw = 0.0f;
    scene.objects.push_back(box);

    LidarConfig cfg = default_lidar_cfg();
    RayCaster caster(cfg);
    PointCloud cloud_with    = caster.cast(scene);

    scene.objects.clear();
    PointCloud cloud_without = caster.cast(scene);

    EXPECT_NE(cloud_with.points.size(), cloud_without.points.size());
}
