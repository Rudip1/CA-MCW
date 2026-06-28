// test/unit/perception/features/test_channel_gcf_rosette.cpp — Phase 5.

#include <gtest/gtest.h>

#include "vf_robot_controller/perception/common/types.hpp"
#include "vf_robot_controller/perception/features/channels/channel_gcf_rosette.hpp"

using vf_robot_controller::perception::GcfRosetteChannel;
using vf_robot_controller::perception::PerceptionState;

TEST(ChannelGcfRosette, NameAndDim) {
  GcfRosetteChannel ch;
  EXPECT_EQ(ch.name(), "gcf_rosette");
  EXPECT_EQ(ch.dim(), 48);
}

TEST(ChannelGcfRosette, ZerosWhenNoData) {
  GcfRosetteChannel ch;
  PerceptionState s;  // gcf_fresh=false, no composite
  Eigen::VectorXf out = Eigen::VectorXf::Constant(48, -99.0f);
  ch.compute(s, out);
  for (int i = 0; i < 48; ++i) EXPECT_FLOAT_EQ(out(i), 0.0f);
}

TEST(ChannelGcfRosette, FallbackBroadcastsScalar) {
  GcfRosetteChannel ch;
  PerceptionState s;
  s.gcf_scalar = 0.7f;
  s.gcf_fresh = true;
  // No composite — should broadcast scalar onto the 16 composite slots
  // (every 3rd dim) and leave clearance/clutter as zero.
  Eigen::VectorXf out(48);
  ch.compute(s, out);
  for (int i = 0; i < 16; ++i) {
    EXPECT_NEAR(out(i * 3 + 0), 0.7f, 1e-6f);
    EXPECT_FLOAT_EQ(out(i * 3 + 1), 0.0f);
    EXPECT_FLOAT_EQ(out(i * 3 + 2), 0.0f);
  }
}
