#include <gtest/gtest.h>
#include "lidar_gen/annotation_generator.hpp"
#include "lidar_gen/types.hpp"
#include "lidar_gen/scene.hpp"

using namespace lidar_gen;

TEST(AnnotationTest, PointLabelsMatchPointCount) {
    PointCloud cloud;
    for (int i = 0; i < 50; ++i) {
        Point3D p; p.x = static_cast<float>(i); p.y = 0; p.z = 0; p.intensity = 0.5f;
        cloud.points.push_back(p);
    }
    SceneDescription scene;
    SceneAnnotation ann = generate_annotations(cloud, scene);
    EXPECT_EQ(ann.point_labels.size(), cloud.points.size());
}

TEST(AnnotationTest, BoxCountMatchesObjects) {
    PointCloud cloud;
    SceneDescription scene;
    SceneObject obj;
    obj.class_name = "Car";
    obj.cx = 5; obj.cy = 0; obj.cz = 0;
    obj.length = 4; obj.width = 2; obj.height = 1.5f; obj.yaw = 0;
    scene.objects.push_back(obj);

    SceneAnnotation ann = generate_annotations(cloud, scene);
    EXPECT_EQ(ann.boxes.size(), 1u);
    EXPECT_EQ(ann.boxes[0].class_name, "Car");
}
