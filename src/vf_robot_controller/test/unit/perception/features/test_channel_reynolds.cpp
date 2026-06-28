// test/unit/perception/features/test_channel_reynolds.cpp — Phase 5.

#include <gtest/gtest.h>

#include "vf_robot_controller/perception/common/types.hpp"
#include "vf_robot_controller/perception/features/channels/channel_reynolds.hpp"

using vf_robot_controller::perception::PathPoint;
using vf_robot_controller::perception::PerceptionState;
using vf_robot_controller::perception::ReynoldsChannel;

TEST(ChannelReynolds, NameAndDim) {
  ReynoldsChannel ch;
  EXPECT_EQ(ch.name(), "reynolds");
  EXPECT_EQ(ch.dim(), 4);
}

TEST(ChannelReynolds, ValuesStayInUnitInterval) {
  ReynoldsChannel ch;
  PerceptionState s;
  s.gcf_scalar = 0.4f;
  s.distance_to_goal = 2.5f;
  for (int i = 0; i < 10; ++i) {
    s.path.push_back({0.5f * static_cast<float>(i), 0.0f});
  }
  Eigen::VectorXf out(4);
  ch.compute(s, out);
  for (int i = 0; i < 4; ++i) {
    EXPECT_GE(out(i), 0.0f);
    EXPECT_LE(out(i), 1.0f);
  }
}

TEST(ChannelReynolds, GoalSeekingSaturatesNearGoal) {
  ReynoldsChannel ch;
  PerceptionState s;
  s.distance_to_goal = 0.0f;  // at goal → cohesion-to-goal == 1
  Eigen::VectorXf out(4);
  ch.compute(s, out);
  EXPECT_NEAR(out(3), 1.0f, 1e-6f);
}

TEST(ChannelReynolds, GoalSeekingZeroFarFromGoal) {
  ReynoldsChannel ch;
  PerceptionState s;
  s.distance_to_goal = 50.0f;
  Eigen::VectorXf out(4);
  ch.compute(s, out);
  EXPECT_NEAR(out(3), 0.0f, 1e-6f);
}

TEST(ChannelReynolds, AlignmentOneWhenHeadingMatchesPath) {
  ReynoldsChannel ch;
  PerceptionState s;
  s.path.push_back({0.0f, 0.0f});
  s.path.push_back({1.0f, 0.0f});
  s.path.push_back({2.0f, 0.0f});
  s.robot_pose.x = 0.0;
  s.robot_pose.y = 0.0;
  s.robot_pose.theta = 0.0;  // heading matches +x path
  Eigen::VectorXf out(4);
  ch.compute(s, out);
  // alignment = (cos(0)*0.5 + 0.5) == 1.0
  EXPECT_NEAR(out(1), 1.0f, 1e-5f);
}
