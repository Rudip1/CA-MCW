// test/unit/meta_critic/test_imitation_velocity_provider.cpp - Phase 8.
//
// Validates the staleness + zero-twist semantics of ImitationVelocityProvider.
// Real ROS subscription wiring is covered by the live Gazebo acceptance run;
// these tests exercise the cache + fallback contract directly.

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include "vf_robot_controller/meta_critic/imitation_velocity_provider.hpp"

using vf_robot_controller::meta_critic::ImitationVelocityProvider;

class ImitationVelocityProviderTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!rclcpp::ok()) rclcpp::init(0, nullptr);
    node_ = std::make_shared<rclcpp_lifecycle::LifecycleNode>("ivp_test_node");
  }
  std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node_;
};

TEST_F(ImitationVelocityProviderTest, NoMessageReturnsZeroTwistAndNotOk) {
  ImitationVelocityProvider p;
  p.configure(node_, "FollowPath");
  auto [twist, ok] = p.getCommand();
  EXPECT_FALSE(ok);
  EXPECT_FLOAT_EQ(twist.linear.x, 0.0f);
  EXPECT_FLOAT_EQ(twist.angular.z, 0.0f);
}

TEST_F(ImitationVelocityProviderTest, FreshInjectedMessageReturnedOk) {
  ImitationVelocityProvider p;
  p.configure(node_, "FollowPath");
  p.injectMessageForTest(0.21, 0.05);
  auto [twist, ok] = p.getCommand();
  EXPECT_TRUE(ok);
  EXPECT_NEAR(twist.linear.x, 0.21, 1e-6);
  EXPECT_NEAR(twist.angular.z, 0.05, 1e-6);
}

TEST_F(ImitationVelocityProviderTest, ShortTimeoutTriggersStaleFallback) {
  // Set a very short timeout so the next sleep makes the cached value stale.
  node_->declare_parameter<int>("FollowPath.imitation_timeout_ms", 50);
  ImitationVelocityProvider p;
  p.configure(node_, "FollowPath");
  p.injectMessageForTest(0.30, 1.0);
  std::this_thread::sleep_for(std::chrono::milliseconds(120));
  auto [twist, ok] = p.getCommand();
  EXPECT_FALSE(ok);
  EXPECT_FLOAT_EQ(twist.linear.x, 0.0f);
  EXPECT_FLOAT_EQ(twist.angular.z, 0.0f);
}

TEST_F(ImitationVelocityProviderTest, MultipleInjectionsKeepLatest) {
  ImitationVelocityProvider p;
  p.configure(node_, "FollowPath");
  p.injectMessageForTest(0.10, 0.0);
  p.injectMessageForTest(0.20, 0.5);
  p.injectMessageForTest(0.05, -0.5);
  auto [twist, ok] = p.getCommand();
  EXPECT_TRUE(ok);
  EXPECT_NEAR(twist.linear.x, 0.05, 1e-6);
  EXPECT_NEAR(twist.angular.z, -0.5, 1e-6);
}
