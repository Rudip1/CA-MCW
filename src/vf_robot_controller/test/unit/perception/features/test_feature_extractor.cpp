// test/unit/perception/features/test_feature_extractor.cpp — Phase 5.

#include <gtest/gtest.h>

#include "vf_robot_controller/perception/common/types.hpp"
#include "vf_robot_controller/perception/features/feature_extractor.hpp"

using vf_robot_controller::perception::FeatureExtractor;
using vf_robot_controller::perception::makeChannel;
using vf_robot_controller::perception::PerceptionState;

TEST(FeatureExtractor, EmptyByDefault) {
  FeatureExtractor fx;
  EXPECT_EQ(fx.totalDim(), 0);
  PerceptionState s;
  auto v = fx.extract(s);
  EXPECT_EQ(v.size(), 0);
}

TEST(FeatureExtractor, FactoryKnowsAllPhase5Channels) {
  EXPECT_NE(makeChannel("robot_state"), nullptr);
  EXPECT_NE(makeChannel("context"), nullptr);
  EXPECT_NE(makeChannel("path_geometry"), nullptr);
  EXPECT_NE(makeChannel("gcf_rosette"), nullptr);
  EXPECT_NE(makeChannel("critic_history"), nullptr);
  EXPECT_NE(makeChannel("obstacle_dynamics"), nullptr);
  EXPECT_NE(makeChannel("reynolds"), nullptr);
  EXPECT_NE(makeChannel("slam_persistent"), nullptr);
  EXPECT_EQ(makeChannel("nonsense"), nullptr);
}

TEST(FeatureExtractor, TotalDimSumsChannels) {
  FeatureExtractor fx;
  fx.addChannel(makeChannel("robot_state"));   // 9
  fx.addChannel(makeChannel("context"));        // 9
  fx.addChannel(makeChannel("path_geometry"));  // 14
  EXPECT_EQ(fx.totalDim(), 9 + 9 + 14);
}

TEST(FeatureExtractor, ExtractFillsEachChannelInOrder) {
  FeatureExtractor fx;
  fx.addChannel(makeChannel("robot_state"));   // 9
  fx.addChannel(makeChannel("context"));        // 9 (one-hot id 2)

  PerceptionState s;
  s.velocity.x() = 1.5f;
  s.context_id = 2;

  auto v = fx.extract(s);
  ASSERT_EQ(v.size(), 18);
  EXPECT_FLOAT_EQ(v(0), 1.5f);            // robot_state vx
  EXPECT_FLOAT_EQ(v(9 + 2), 1.0f);        // context one-hot
}

TEST(FeatureExtractor, V1ChannelsTotalSize) {
  FeatureExtractor fx;
  for (const auto & n : {"robot_state", "context", "path_geometry",
                          "gcf_rosette", "critic_history", "obstacle_dynamics"}) {
    fx.addChannel(makeChannel(n));
  }
  // 9 + 9 + 14 + 48 + 30 + 16 = 126
  EXPECT_EQ(fx.totalDim(), 126);
}

TEST(FeatureExtractor, ChannelMetadataMatches) {
  FeatureExtractor fx;
  fx.addChannel(makeChannel("robot_state"));
  fx.addChannel(makeChannel("path_geometry"));
  auto names = fx.channelNames();
  auto dims = fx.channelDims();
  ASSERT_EQ(names.size(), 2u);
  EXPECT_EQ(names[0], "robot_state");
  EXPECT_EQ(names[1], "path_geometry");
  EXPECT_EQ(dims[0], 9);
  EXPECT_EQ(dims[1], 14);
}
