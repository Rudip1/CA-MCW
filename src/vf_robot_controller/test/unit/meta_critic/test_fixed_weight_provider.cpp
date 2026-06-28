// test/unit/meta_critic/test_fixed_weight_provider.cpp — Phase 2.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include "vf_robot_controller/meta_critic/fixed_weight_provider.hpp"

using vf_robot_controller::meta_critic::FixedWeightProvider;

class FixedWeightProviderTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!rclcpp::ok()) rclcpp::init(0, nullptr);
    node_ = std::make_shared<rclcpp_lifecycle::LifecycleNode>("fwp_test_node");
  }
  std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node_;
};

TEST_F(FixedWeightProviderTest, NameIsFixed) {
  FixedWeightProvider p;
  EXPECT_EQ(p.name(), "fixed");
}

TEST_F(FixedWeightProviderTest, NoYamlDefaultsToOnes) {
  FixedWeightProvider p;
  p.configure(node_, "FollowPath", 5);
  EXPECT_EQ(p.numCritics(), 5);
  Eigen::VectorXf feat;
  auto w = p.getWeights(feat);
  ASSERT_EQ(w.size(), 5u);
  for (float v : w) EXPECT_FLOAT_EQ(v, 1.0f);
}

TEST_F(FixedWeightProviderTest, ReadsYamlVector) {
  node_->declare_parameter<std::vector<double>>(
    "FollowPath.fixed_weights", {0.5, 1.0, 2.0, 4.0});
  FixedWeightProvider p;
  p.configure(node_, "FollowPath", 4);
  Eigen::VectorXf feat;
  auto w = p.getWeights(feat);
  ASSERT_EQ(w.size(), 4u);
  EXPECT_FLOAT_EQ(w[0], 0.5f);
  EXPECT_FLOAT_EQ(w[1], 1.0f);
  EXPECT_FLOAT_EQ(w[2], 2.0f);
  EXPECT_FLOAT_EQ(w[3], 4.0f);
}

TEST_F(FixedWeightProviderTest, ShortVectorPadsWithOnes) {
  node_->declare_parameter<std::vector<double>>(
    "FollowPath.fixed_weights", {0.5, 2.0});
  FixedWeightProvider p;
  p.configure(node_, "FollowPath", 4);
  Eigen::VectorXf feat;
  auto w = p.getWeights(feat);
  ASSERT_EQ(w.size(), 4u);
  EXPECT_FLOAT_EQ(w[0], 0.5f);
  EXPECT_FLOAT_EQ(w[1], 2.0f);
  EXPECT_FLOAT_EQ(w[2], 1.0f);
  EXPECT_FLOAT_EQ(w[3], 1.0f);
}

TEST_F(FixedWeightProviderTest, FeaturesAreIgnored) {
  FixedWeightProvider p;
  p.setWeightsForTest({1.0f, 2.0f, 3.0f});
  Eigen::VectorXf feat_a = Eigen::VectorXf::Zero(10);
  Eigen::VectorXf feat_b = Eigen::VectorXf::Random(50);
  EXPECT_EQ(p.getWeights(feat_a), p.getWeights(feat_b));
}
