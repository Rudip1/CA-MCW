// test/unit/perception/features/test_channel_obstacle_dynamics.cpp — Phase 5.

#include <gtest/gtest.h>

#include <memory>

#include <nav2_costmap_2d/costmap_2d.hpp>

#include "vf_robot_controller/perception/common/types.hpp"
#include "vf_robot_controller/perception/features/channels/channel_obstacle_dynamics.hpp"

using nav2_costmap_2d::Costmap2D;
using vf_robot_controller::perception::ObstacleDynamicsChannel;
using vf_robot_controller::perception::PerceptionState;

namespace {
std::shared_ptr<Costmap2D> makeFlatCostmap(int sx, int sy, double res, uint8_t fill)
{
  // origin at (-sx*res/2, -sy*res/2) so robot at (0,0) lands at the centre.
  auto cm = std::make_shared<Costmap2D>(sx, sy, res,
                                        -sx * res * 0.5,
                                        -sy * res * 0.5,
                                        fill);
  return cm;
}
}  // namespace

TEST(ChannelObstacleDynamics, NameAndDim) {
  ObstacleDynamicsChannel ch;
  EXPECT_EQ(ch.name(), "obstacle_dynamics");
  EXPECT_EQ(ch.dim(), 16);
}

TEST(ChannelObstacleDynamics, ZerosWhenSnapshotsMissing) {
  ObstacleDynamicsChannel ch;
  PerceptionState s;  // both costmaps null
  Eigen::VectorXf out = Eigen::VectorXf::Constant(16, -99.0f);
  ch.compute(s, out);
  EXPECT_TRUE(out.isZero());
}

TEST(ChannelObstacleDynamics, ZerosWhenNoChange) {
  ObstacleDynamicsChannel ch;
  PerceptionState s;
  s.costmap_now = makeFlatCostmap(40, 40, 0.1, 0);
  s.costmap_prev = makeFlatCostmap(40, 40, 0.1, 0);
  Eigen::VectorXf out(16);
  ch.compute(s, out);
  EXPECT_TRUE(out.isZero());
}

TEST(ChannelObstacleDynamics, PositiveDeltaShowsUpInSomeBin) {
  ObstacleDynamicsChannel ch;
  PerceptionState s;
  auto now = makeFlatCostmap(40, 40, 0.1, 0);
  auto prev = makeFlatCostmap(40, 40, 0.1, 0);
  // Place a single obstacle 1 m ahead of (0,0) in the now snapshot.
  unsigned int mx = 0, my = 0;
  ASSERT_TRUE(now->worldToMap(1.0, 0.0, mx, my));
  now->setCost(mx, my, 254);
  s.costmap_now = now;
  s.costmap_prev = prev;
  Eigen::VectorXf out(16);
  ch.compute(s, out);
  // At least one positive bin should be > 0.
  bool any_pos = false;
  for (int i = 0; i < 8; ++i) {
    if (out(i * 2 + 0) > 0.0f) any_pos = true;
  }
  EXPECT_TRUE(any_pos);
  // No negative deltas anywhere.
  for (int i = 0; i < 8; ++i) {
    EXPECT_LE(out(i * 2 + 1), 0.0f);
  }
}
