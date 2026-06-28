// ImitationVelocityProvider - Phase 8.
//
// Subscribes to a Twist topic published by the Python imitation inference
// sidecar (default /vf_controller/imitation_cmd_vel) and caches the latest
// message with a timestamp for staleness checks. The controller, in
// IMITATION mode, calls getCommand() per cycle to retrieve a network-
// predicted (vx, wz) and bypasses MPPI entirely.

#include "vf_robot_controller/meta_critic/imitation_velocity_provider.hpp"

#include <algorithm>

#include <nav2_util/node_utils.hpp>
#include <rclcpp/qos.hpp>

namespace vf_robot_controller::meta_critic {

ImitationVelocityProvider::ImitationVelocityProvider() = default;

void ImitationVelocityProvider::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & node_weak,
  const std::string & param_ns)
{
  node_weak_ = node_weak;
  auto node = node_weak.lock();
  if (!node) return;
  logger_ = node->get_logger();

  std::string topic = "/vf_controller/imitation_cmd_vel";
  int timeout_ms = 200;
  nav2_util::declare_parameter_if_not_declared(
    node, param_ns + ".imitation_topic", rclcpp::ParameterValue(topic));
  nav2_util::declare_parameter_if_not_declared(
    node, param_ns + ".imitation_timeout_ms",
    rclcpp::ParameterValue(timeout_ms));
  node->get_parameter(param_ns + ".imitation_topic", topic);
  node->get_parameter(param_ns + ".imitation_timeout_ms", timeout_ms);
  timeout_ = std::chrono::milliseconds(std::max(50, timeout_ms));

  rclcpp::QoS qos(rclcpp::KeepLast(5));
  qos.best_effort();
  sub_ = node->create_subscription<geometry_msgs::msg::Twist>(
    topic, qos,
    [this](geometry_msgs::msg::Twist::SharedPtr m) { onTwistMsg(m); });

  RCLCPP_INFO(
    logger_,
    "ImitationVelocityProvider: subscribed to '%s' (timeout=%d ms)",
    topic.c_str(), timeout_ms);
}

void ImitationVelocityProvider::onTwistMsg(
  const geometry_msgs::msg::Twist::SharedPtr msg)
{
  std::lock_guard<std::mutex> g(mu_);
  latest_ = *msg;
  if (auto node = node_weak_.lock()) {
    latest_stamp_ = node->now();
  }
  has_message_ = true;
  warned_stale_once_ = false;
}

std::pair<geometry_msgs::msg::Twist, bool>
ImitationVelocityProvider::getCommand()
{
  std::lock_guard<std::mutex> g(mu_);
  geometry_msgs::msg::Twist zero;
  if (!has_message_) {
    if (!warned_stale_once_) {
      RCLCPP_WARN(
        logger_,
        "ImitationVelocityProvider: no message yet on imitation topic; "
        "returning zero twist.");
      warned_stale_once_ = true;
    }
    return {zero, false};
  }
  auto node = node_weak_.lock();
  if (node) {
    const auto age = node->now() - latest_stamp_;
    if (age > rclcpp::Duration(timeout_)) {
      if (!warned_stale_once_) {
        RCLCPP_WARN(
          logger_,
          "ImitationVelocityProvider: stale (>%ld ms); returning zero twist.",
          static_cast<long>(timeout_.count()));
        warned_stale_once_ = true;
      }
      return {zero, false};
    }
  }
  return {latest_, true};
}

void ImitationVelocityProvider::injectMessageForTest(double vx, double wz)
{
  std::lock_guard<std::mutex> g(mu_);
  latest_.linear.x = vx;
  latest_.angular.z = wz;
  has_message_ = true;
  if (auto node = node_weak_.lock()) {
    latest_stamp_ = node->now();
  }
}

}  // namespace vf_robot_controller::meta_critic
