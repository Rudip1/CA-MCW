// test/unit/perception/features/test_channel_context.cpp — Phase 5.

#include <gtest/gtest.h>

#include "vf_robot_controller/perception/common/types.hpp"
#include "vf_robot_controller/perception/features/channels/channel_context.hpp"

using vf_robot_controller::perception::ContextChannel;
using vf_robot_controller::perception::PerceptionState;

TEST(ChannelContext, NameAndDim) {
  ContextChannel ch;
  EXPECT_EQ(ch.name(), "context");
  EXPECT_EQ(ch.dim(), 9);
}

TEST(ChannelContext, OneHotForKnownContext) {
  ContextChannel ch;
  PerceptionState s;
  s.context_id = 3;  // DYNAMIC
  Eigen::VectorXf out = Eigen::VectorXf::Constant(9, -99.0f);
  ch.compute(s, out);
  for (int i = 0; i < 9; ++i) {
    if (i == 3) EXPECT_FLOAT_EQ(out(i), 1.0f);
    else        EXPECT_FLOAT_EQ(out(i), 0.0f);
  }
}

TEST(ChannelContext, UnknownYieldsAllZero) {
  ContextChannel ch;
  PerceptionState s;  // default context_id == 255 (UNKNOWN)
  Eigen::VectorXf out(9);
  ch.compute(s, out);
  for (int i = 0; i < 9; ++i) EXPECT_FLOAT_EQ(out(i), 0.0f);
}

TEST(ChannelContext, OutOfRangeIdYieldsAllZero) {
  ContextChannel ch;
  PerceptionState s;
  s.context_id = 12;  // beyond reserved 9 slots
  Eigen::VectorXf out(9);
  ch.compute(s, out);
  for (int i = 0; i < 9; ++i) EXPECT_FLOAT_EQ(out(i), 0.0f);
}
