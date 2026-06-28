// test/unit/perception/features/test_channel_robot_state.cpp — Phase 5.

#include <gtest/gtest.h>

#include <cmath>

#include "vf_robot_controller/perception/common/types.hpp"
#include "vf_robot_controller/perception/features/channels/channel_robot_state.hpp"

using vf_robot_controller::perception::PerceptionState;
using vf_robot_controller::perception::RobotStateChannel;

TEST(ChannelRobotState, NameAndDim) {
  RobotStateChannel ch;
  EXPECT_EQ(ch.name(), "robot_state");
  EXPECT_EQ(ch.dim(), 9);
}

TEST(ChannelRobotState, FillsAllNineSlots) {
  RobotStateChannel ch;
  PerceptionState s;
  s.velocity = Eigen::Vector3f(0.5f, 0.0f, 0.2f);
  s.acceleration = Eigen::Vector3f(0.1f, 0.0f, 0.05f);
  s.robot_pose.theta = M_PI / 2.0;

  Eigen::VectorXf out = Eigen::VectorXf::Constant(9, -99.0f);
  ch.compute(s, out);
  EXPECT_FLOAT_EQ(out(0), 0.5f);
  EXPECT_FLOAT_EQ(out(1), 0.0f);
  EXPECT_FLOAT_EQ(out(2), 0.2f);
  EXPECT_NEAR(out(3), 1.0f, 1e-5f);          // sin(pi/2)
  EXPECT_NEAR(out(4), 0.0f, 1e-5f);          // cos(pi/2)
  EXPECT_NEAR(out(5), 0.5f, 1e-5f);          // |v|
  EXPECT_FLOAT_EQ(out(6), 0.1f);
  EXPECT_FLOAT_EQ(out(7), 0.0f);
  EXPECT_FLOAT_EQ(out(8), 0.05f);
}

TEST(ChannelRobotState, ZerosForDefaultState) {
  RobotStateChannel ch;
  PerceptionState s;  // default: zero velocity, theta=0
  Eigen::VectorXf out(9);
  ch.compute(s, out);
  EXPECT_FLOAT_EQ(out(0), 0.0f);
  EXPECT_FLOAT_EQ(out(3), 0.0f);  // sin(0)
  EXPECT_FLOAT_EQ(out(4), 1.0f);  // cos(0)
  EXPECT_FLOAT_EQ(out(5), 0.0f);  // |v|
}
