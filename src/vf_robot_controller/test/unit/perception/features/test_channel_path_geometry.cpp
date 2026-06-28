// test/unit/perception/features/test_channel_path_geometry.cpp — Phase 5.

#include <gtest/gtest.h>

#include <cmath>

#include "vf_robot_controller/perception/common/types.hpp"
#include "vf_robot_controller/perception/features/channels/channel_path_geometry.hpp"

using vf_robot_controller::perception::PathGeometryChannel;
using vf_robot_controller::perception::PathPoint;
using vf_robot_controller::perception::PerceptionState;

TEST(ChannelPathGeometry, NameAndDim) {
  PathGeometryChannel ch;
  EXPECT_EQ(ch.name(), "path_geometry");
  EXPECT_EQ(ch.dim(), 14);
}

TEST(ChannelPathGeometry, EmptyPathFlagsLastDim) {
  PathGeometryChannel ch;
  PerceptionState s;
  Eigen::VectorXf out(14);
  ch.compute(s, out);
  EXPECT_FLOAT_EQ(out(13), 1.0f);  // path empty/stale flag
  for (int i = 0; i < 13; ++i) EXPECT_FLOAT_EQ(out(i), 0.0f);
}

TEST(ChannelPathGeometry, StraightPathAlongX) {
  PathGeometryChannel ch;
  PerceptionState s;
  s.path.reserve(20);
  for (int i = 0; i < 20; ++i) {
    s.path.push_back({0.5f * static_cast<float>(i), 0.0f});
  }
  s.robot_pose.x = 0.0;
  s.robot_pose.y = 0.0;
  s.robot_pose.theta = 0.0;
  s.distance_to_goal = 9.5f;  // 19 * 0.5

  Eigen::VectorXf out(14);
  ch.compute(s, out);
  EXPECT_NEAR(out(0), 9.5f, 0.01f);                     // distance_to_goal
  EXPECT_GT(out(1), 9.0f);                              // remaining arc ≈ 9.5
  EXPECT_NEAR(out(2), 0.0f, 0.01f);                    // cross-track ≈ 0
  EXPECT_FLOAT_EQ(out(13), 0.0f);                      // not stale
}

TEST(ChannelPathGeometry, LateralOffsetGivesNonZeroCrossTrack) {
  PathGeometryChannel ch;
  PerceptionState s;
  s.path.reserve(20);
  for (int i = 0; i < 20; ++i) {
    s.path.push_back({0.5f * static_cast<float>(i), 0.0f});
  }
  s.robot_pose.x = 0.0;
  s.robot_pose.y = 0.5;  // 0.5 m above the +x path

  Eigen::VectorXf out(14);
  ch.compute(s, out);
  EXPECT_NEAR(std::abs(out(2)), 0.5f, 0.05f);
}
