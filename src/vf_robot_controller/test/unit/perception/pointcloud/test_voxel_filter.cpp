// test/unit/perception/pointcloud/test_voxel_filter.cpp — Phase 4.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "vf_robot_controller/perception/pointcloud/voxel_filter.hpp"

using vf_robot_controller::perception::VoxelFilter;

namespace {

// Build a PointCloud2 from a flat (x,y,z) vector.
sensor_msgs::msg::PointCloud2 makeCloud(const std::vector<std::array<float, 3>> & pts)
{
  sensor_msgs::msg::PointCloud2 c;
  c.header.frame_id = "test";
  c.height = 1;
  c.width = static_cast<uint32_t>(pts.size());
  c.is_bigendian = false;
  c.is_dense = false;

  sensor_msgs::PointCloud2Modifier mod(c);
  mod.setPointCloud2FieldsByString(1, "xyz");
  mod.resize(pts.size());

  sensor_msgs::PointCloud2Iterator<float> ix(c, "x");
  sensor_msgs::PointCloud2Iterator<float> iy(c, "y");
  sensor_msgs::PointCloud2Iterator<float> iz(c, "z");
  for (const auto & p : pts) {
    *ix = p[0]; *iy = p[1]; *iz = p[2];
    ++ix; ++iy; ++iz;
  }
  return c;
}

}  // namespace

TEST(VoxelFilter, EmptyCloudYieldsEmpty) {
  VoxelFilter f(0.05f);
  auto in = makeCloud({});
  std::vector<std::array<float, 3>> out;
  EXPECT_EQ(f.filter(in, out), 0u);
  EXPECT_TRUE(out.empty());
}

TEST(VoxelFilter, IdenticalPointsCollapseToOneCentroid) {
  VoxelFilter f(0.10f);
  std::vector<std::array<float, 3>> pts(500, {1.0f, 1.0f, 0.5f});
  auto in = makeCloud(pts);
  std::vector<std::array<float, 3>> out;
  EXPECT_EQ(f.filter(in, out), 1u);
  EXPECT_NEAR(out[0][0], 1.0f, 1e-3f);
  EXPECT_NEAR(out[0][1], 1.0f, 1e-3f);
  EXPECT_NEAR(out[0][2], 0.5f, 1e-3f);
}

TEST(VoxelFilter, GridGivesOnePointPerCell) {
  VoxelFilter f(0.5f);
  // 4×4×1 = 16 points at half-leaf spacing — each lands in its own voxel.
  std::vector<std::array<float, 3>> pts;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      pts.push_back({i * 1.0f, j * 1.0f, 0.0f});
    }
  }
  auto in = makeCloud(pts);
  std::vector<std::array<float, 3>> out;
  EXPECT_EQ(f.filter(in, out), 16u);
}

TEST(VoxelFilter, DownsamplingReducesCount) {
  VoxelFilter f(1.0f);
  // 1000 points scattered in a single 1x1x1 box → all collapse to ≤ 1 cell.
  std::vector<std::array<float, 3>> pts;
  pts.reserve(1000);
  for (int i = 0; i < 1000; ++i) {
    pts.push_back({(i % 10) * 0.05f, ((i / 10) % 10) * 0.05f, 0.0f});
  }
  auto in = makeCloud(pts);
  std::vector<std::array<float, 3>> out;
  const auto n = f.filter(in, out);
  EXPECT_LT(n, pts.size());
  EXPECT_LE(n, 4u);  // 0.5 m spread / 1.0 m leaf — at most a few voxels.
}

TEST(VoxelFilter, NaNAndInfFiltered) {
  VoxelFilter f(0.05f);
  std::vector<std::array<float, 3>> pts = {
    {1.0f, 1.0f, 0.0f},
    {std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f},
    {std::numeric_limits<float>::infinity(), 0.0f, 0.0f},
    {2.0f, 2.0f, 0.0f},
  };
  auto in = makeCloud(pts);
  std::vector<std::array<float, 3>> out;
  EXPECT_EQ(f.filter(in, out), 2u);
}

TEST(VoxelFilter, OutputCloudHasInputHeader) {
  VoxelFilter f(0.05f);
  auto in = makeCloud({{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}});
  in.header.frame_id = "odom_test";
  sensor_msgs::msg::PointCloud2 out;
  f.filter(in, out);
  EXPECT_EQ(out.header.frame_id, "odom_test");
  EXPECT_EQ(out.width, 2u);
}
