// test/unit/perception/features/test_channel_critic_history.cpp — Phase 5.

#include <gtest/gtest.h>

#include <cmath>

#include "vf_robot_controller/perception/common/types.hpp"
#include "vf_robot_controller/perception/features/channels/channel_critic_history.hpp"

using vf_robot_controller::perception::CriticCostSample;
using vf_robot_controller::perception::CriticHistoryChannel;
using vf_robot_controller::perception::PerceptionState;

TEST(ChannelCriticHistory, NameAndDim) {
  CriticHistoryChannel ch;
  EXPECT_EQ(ch.name(), "critic_history");
  EXPECT_EQ(ch.dim(), 30);
}

TEST(ChannelCriticHistory, ZerosWhenHistoryEmpty) {
  CriticHistoryChannel ch;
  PerceptionState s;
  Eigen::VectorXf out = Eigen::VectorXf::Constant(30, -99.0f);
  ch.compute(s, out);
  EXPECT_TRUE(out.isZero());
}

TEST(ChannelCriticHistory, NewestSampleLandsInLastSlot) {
  CriticHistoryChannel ch;
  PerceptionState s;
  // Single sample with deterministic costs: mean=2, max=3, std=sqrt(2/3)
  CriticCostSample one;
  one.costs = {1.0f, 2.0f, 3.0f};
  s.critic_history.push_back(one);
  Eigen::VectorXf out(30);
  ch.compute(s, out);
  // First 27 entries (9 unused cycles × 3) should be zero
  for (int i = 0; i < 27; ++i) EXPECT_FLOAT_EQ(out(i), 0.0f);
  // Last 3 entries are the single sample's stats
  EXPECT_NEAR(out(27), 2.0f, 1e-5f);                          // mean
  EXPECT_NEAR(out(28), 3.0f, 1e-5f);                          // max
  EXPECT_NEAR(out(29), std::sqrt(2.0f / 3.0f), 1e-5f);        // std (population)
}

TEST(ChannelCriticHistory, EmptyCostsVectorYieldsZeroSlot) {
  CriticHistoryChannel ch;
  PerceptionState s;
  CriticCostSample empty;
  s.critic_history.push_back(empty);
  Eigen::VectorXf out(30);
  ch.compute(s, out);
  EXPECT_TRUE(out.isZero());
}

TEST(ChannelCriticHistory, OldestSampleClippedWhenOver10) {
  CriticHistoryChannel ch;
  PerceptionState s;
  for (int i = 0; i < 12; ++i) {
    CriticCostSample sample;
    sample.costs = {static_cast<float>(i), static_cast<float>(i)};
    s.critic_history.push_back(sample);
  }
  Eigen::VectorXf out(30);
  ch.compute(s, out);
  // Oldest considered is index 2 (i=2), newest is index 11 (i=11).
  // First slot mean = 2, last slot mean = 11.
  EXPECT_NEAR(out(0), 2.0f, 1e-5f);
  EXPECT_NEAR(out(27), 11.0f, 1e-5f);
}
