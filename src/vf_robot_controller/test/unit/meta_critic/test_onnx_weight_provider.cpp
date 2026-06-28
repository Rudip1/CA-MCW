// test/unit/meta_critic/test_onnx_weight_provider.cpp - Phase 8.
//
// Behavioural tests that work whether or not onnxruntime is built into the
// package. They exercise the fallback path: empty onnx_path, stale
// features, dim mismatch. The "actually run a model" path is covered by
// docs/training.md sanity check + the verify_export pytest.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include "vf_robot_controller/meta_critic/onnx_weight_provider.hpp"

using vf_robot_controller::meta_critic::OnnxWeightProvider;

class OnnxWeightProviderTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!rclcpp::ok()) rclcpp::init(0, nullptr);
    node_ = std::make_shared<rclcpp_lifecycle::LifecycleNode>("owp_test_node");
  }
  std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node_;
};

TEST_F(OnnxWeightProviderTest, NameIsOnnx) {
  OnnxWeightProvider p;
  EXPECT_EQ(p.name(), "onnx");
}

TEST_F(OnnxWeightProviderTest, EmptyOnnxPathFallsBackToOnes) {
  OnnxWeightProvider p;
  p.configure(node_, "FollowPath", 11);
  EXPECT_EQ(p.numCritics(), 11);
  EXPECT_FALSE(p.isModelLoaded());

  Eigen::VectorXf feat;  // empty -> stale path
  auto w = p.getWeights(feat);
  ASSERT_EQ(w.size(), 11u);
  for (float v : w) EXPECT_FLOAT_EQ(v, 1.0f);
}

TEST_F(OnnxWeightProviderTest, FallbackVectorIsRespected) {
  node_->declare_parameter<std::vector<double>>(
    "FollowPath.fixed_weights",
    {0.5, 1.0, 1.5, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
  OnnxWeightProvider p;
  p.configure(node_, "FollowPath", 11);

  Eigen::VectorXf feat;
  auto w = p.getWeights(feat);
  ASSERT_EQ(w.size(), 11u);
  EXPECT_FLOAT_EQ(w[0], 0.5f);
  EXPECT_FLOAT_EQ(w[1], 1.0f);
  EXPECT_FLOAT_EQ(w[2], 1.5f);
  EXPECT_FLOAT_EQ(w[3], 2.0f);
}

TEST_F(OnnxWeightProviderTest, NonsenseOnnxPathFallsBackGracefully) {
  node_->declare_parameter<std::string>(
    "FollowPath.onnx_path", "/this/path/does/not/exist.onnx");
  OnnxWeightProvider p;
  p.configure(node_, "FollowPath", 11);
  // No crash, no exception thrown, fallback returns ones.
  EXPECT_FALSE(p.isModelLoaded());
  Eigen::VectorXf feat;
  auto w = p.getWeights(feat);
  ASSERT_EQ(w.size(), 11u);
}

TEST_F(OnnxWeightProviderTest, FeatureDimMismatchUsesFallback) {
  OnnxWeightProvider p;
  p.configure(node_, "FollowPath", 11);
  // Wrong-size features (170 expected by default).
  Eigen::VectorXf feat = Eigen::VectorXf::Zero(7);
  auto w = p.getWeights(feat);
  ASSERT_EQ(w.size(), 11u);
  for (float v : w) EXPECT_FLOAT_EQ(v, 1.0f);
}
