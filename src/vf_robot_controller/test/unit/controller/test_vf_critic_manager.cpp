// test/unit/controller/test_vf_critic_manager.cpp — Phase 2.
//
// Tests the WeightCache singleton + VFCriticManager push pipeline. We can't
// easily exercise the Weighted* wrappers without a full Nav2 controller
// stack, so this file focuses on the cache and manager logic in isolation.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include "vf_robot_controller/controller/vf_critic_manager.hpp"
#include "vf_robot_controller/controller/weight_cache.hpp"
#include "vf_robot_controller/meta_critic/fixed_weight_provider.hpp"

using vf_robot_controller::VFCriticManager;
using vf_robot_controller::WeightCache;
using vf_robot_controller::meta_critic::FixedWeightProvider;

class VFCriticManagerTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!rclcpp::ok()) rclcpp::init(0, nullptr);
    node_ = std::make_shared<rclcpp_lifecycle::LifecycleNode>("vfcm_test_node");
    WeightCache::instance().clear();
  }
  void TearDown() override {
    WeightCache::instance().clear();
  }
  std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node_;
};

TEST_F(VFCriticManagerTest, ConfigurePopulatesKeys) {
  VFCriticManager m;
  m.configure(node_, "FollowPath",
              {"WeightedPathFollowCritic", "WeightedGoalCritic"});
  ASSERT_EQ(m.numCritics(), 2);
  EXPECT_EQ(m.criticKeys()[0], "FollowPath.WeightedPathFollowCritic");
  EXPECT_EQ(m.criticKeys()[1], "FollowPath.WeightedGoalCritic");
}

TEST_F(VFCriticManagerTest, PushWeightsWritesIntoCache) {
  VFCriticManager m;
  m.configure(node_, "FollowPath",
              {"WeightedPathFollowCritic", "WeightedGoalCritic"});

  auto provider = std::make_shared<FixedWeightProvider>();
  provider->setWeightsForTest({2.5f, 0.5f});
  m.setWeightProvider(provider);

  Eigen::VectorXf empty;
  m.pushWeights(empty);

  auto a = WeightCache::instance().getMultiplier("FollowPath.WeightedPathFollowCritic");
  auto b = WeightCache::instance().getMultiplier("FollowPath.WeightedGoalCritic");
  ASSERT_TRUE(a.has_value());
  ASSERT_TRUE(b.has_value());
  EXPECT_FLOAT_EQ(*a, 2.5f);
  EXPECT_FLOAT_EQ(*b, 0.5f);
}

TEST_F(VFCriticManagerTest, EmptyProviderResultDefaultsToOnes) {
  VFCriticManager m;
  m.configure(node_, "FollowPath",
              {"WeightedPathFollowCritic", "WeightedGoalCritic"});

  auto provider = std::make_shared<FixedWeightProvider>();
  provider->setWeightsForTest({});
  m.setWeightProvider(provider);

  Eigen::VectorXf empty;
  m.pushWeights(empty);

  auto a = WeightCache::instance().getMultiplier("FollowPath.WeightedPathFollowCritic");
  ASSERT_TRUE(a.has_value());
  EXPECT_FLOAT_EQ(*a, 1.0f);
}

TEST_F(VFCriticManagerTest, NoProviderIsNoOp) {
  VFCriticManager m;
  m.configure(node_, "FollowPath", {"WeightedPathFollowCritic"});
  Eigen::VectorXf empty;
  m.pushWeights(empty);  // must not crash; cache stays untouched
  EXPECT_FALSE(WeightCache::instance().getMultiplier(
    "FollowPath.WeightedPathFollowCritic").has_value());
}

TEST_F(VFCriticManagerTest, CostCollectionDeltaRoundTrip) {
  VFCriticManager m;
  m.configure(node_, "FollowPath", {"A", "B"});
  m.setCostCollectionActive(true);

  WeightCache::instance().recordDelta("FollowPath.A", {1.0f, 2.0f, 3.0f});
  WeightCache::instance().recordDelta("FollowPath.B", {4.0f});

  auto deltas = m.takeRecordedDeltas();
  EXPECT_EQ(deltas.size(), 2u);
  EXPECT_EQ(deltas["FollowPath.A"], (std::vector<float>{1.0f, 2.0f, 3.0f}));
  EXPECT_EQ(deltas["FollowPath.B"], (std::vector<float>{4.0f}));

  // Second take returns empty (deltas were cleared).
  EXPECT_TRUE(m.takeRecordedDeltas().empty());
  m.setCostCollectionActive(false);
}

TEST_F(VFCriticManagerTest, CacheActiveFlagToggles) {
  auto & c = WeightCache::instance();
  EXPECT_FALSE(c.isActive());
  c.setActive(true);
  EXPECT_TRUE(c.isActive());
  c.setActive(false);
  EXPECT_FALSE(c.isActive());
}
