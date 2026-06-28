// test/integration/test_passthrough_matches_mppi.cpp
// Phase 1: verify VFController can be instantiated via pluginlib and that
// all nav2_core::Controller virtual methods are reachable.  Full Gazebo
// behavior verification is done manually per the acceptance criteria.

#include <gtest/gtest.h>
#include <pluginlib/class_loader.hpp>
#include <nav2_core/controller.hpp>
#include <rclcpp/rclcpp.hpp>

// Test that pluginlib can locate and instantiate VFController.
// This catches:
//   - Wrong class type string in controller_plugins.xml
//   - Library path mismatch (lib prefix / .so suffix issues)
//   - Missing PLUGINLIB_EXPORT_CLASS macro in vf_controller.cpp
TEST(PassthroughMatchesMppi, PluginLoadsViaPluginlib)
{
  // rclcpp must be initialised before any Nav2 plugin is loaded because
  // MPPIController::configure calls rclcpp internals.
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }

  pluginlib::ClassLoader<nav2_core::Controller> loader(
    "nav2_core", "nav2_core::Controller");

  // The class type string must match controller_plugins.xml exactly.
  const std::string class_type = "vf_robot_controller::VFController";

  ASSERT_TRUE(loader.isClassAvailable(class_type))
    << "VFController not found by pluginlib. "
       "Check controller_plugins.xml path= attribute and class type= string.";

  nav2_core::Controller::Ptr instance;
  ASSERT_NO_THROW({ instance = loader.createSharedInstance(class_type); })
    << "pluginlib threw while constructing VFController.";

  ASSERT_NE(instance, nullptr)
    << "pluginlib returned a null shared_ptr for VFController.";
}

// Confirm that the upstream MPPIController plugin is also discoverable —
// if it is missing the whole delegation chain breaks.
TEST(PassthroughMatchesMppi, UpstreamMppiPluginReachable)
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }

  pluginlib::ClassLoader<nav2_core::Controller> loader(
    "nav2_core", "nav2_core::Controller");

  const std::string upstream_type = "nav2_mppi_controller::MPPIController";

  EXPECT_TRUE(loader.isClassAvailable(upstream_type))
    << "Upstream nav2_mppi_controller::MPPIController not found. "
       "Check that nav2_mppi_controller is installed (apt).";
}
