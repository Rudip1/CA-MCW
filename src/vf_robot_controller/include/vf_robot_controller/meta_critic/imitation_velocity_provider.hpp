// ImitationVelocityProvider — Phase 8 (IMITATION mode).
//
// Subscribes to /vf_controller/imitation_cmd_vel (geometry_msgs/Twist) and
// caches the latest message with a staleness check. The C++ controller, when
// running in mode == IMITATION, calls getCommand() per cycle to get the
// network-predicted velocity, bypassing the MPPI optimizer entirely.
//
// This is intentionally NOT an IWeightProvider — INFERENCE and IMITATION are
// separate runtime paths (design anti-pattern #7). They share the same
// dataset and feature pipeline, but at runtime each has its own subscriber,
// its own cache, its own provider class.
//
// Failure semantics (the design notes hard rule: never block the control loop):
//   - No message yet: return zero twist + log once.
//   - Stale (> timeout): return zero twist + log throttled.
//   - The controller decides what to do with zero twist. Velocity smoother
//     downstream will glide toward zero, which is the correct fail-safe.

#ifndef VF_ROBOT_CONTROLLER__META_CRITIC__IMITATION_VELOCITY_PROVIDER_HPP_
#define VF_ROBOT_CONTROLLER__META_CRITIC__IMITATION_VELOCITY_PROVIDER_HPP_

#include <chrono>
#include <memory>
#include <mutex>
#include <string>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

namespace vf_robot_controller::meta_critic {

class ImitationVelocityProvider {
public:
  ImitationVelocityProvider();
  virtual ~ImitationVelocityProvider() = default;

  // Reads:
  //   <param_ns>.imitation_topic            (default "/vf_controller/imitation_cmd_vel")
  //   <param_ns>.imitation_timeout_ms       (default 200)
  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & node,
    const std::string & param_ns);

  // Returns (twist, ok). twist is zero on stale / no message; `ok=false` so
  // the caller can log a one-shot warning, but the twist is always valid.
  std::pair<geometry_msgs::msg::Twist, bool> getCommand();

  // Test seam.
  void injectMessageForTest(double vx, double wz);

private:
  void onTwistMsg(const geometry_msgs::msg::Twist::SharedPtr msg);

  rclcpp_lifecycle::LifecycleNode::WeakPtr node_weak_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_;
  rclcpp::Logger logger_{rclcpp::get_logger("ImitationVelocityProvider")};

  std::mutex mu_;
  geometry_msgs::msg::Twist latest_;
  rclcpp::Time latest_stamp_;
  bool has_message_{false};
  std::chrono::milliseconds timeout_{200};
  bool warned_stale_once_{false};
};

}  // namespace vf_robot_controller::meta_critic

#endif  // VF_ROBOT_CONTROLLER__META_CRITIC__IMITATION_VELOCITY_PROVIDER_HPP_
