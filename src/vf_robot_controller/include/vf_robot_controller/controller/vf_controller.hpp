// VFController — adaptive meta-critic plugin controller for Nav2.
//
// Phase 1: passthrough (delegates to upstream MPPIController).
// Phase 2: adds VFCriticManager + IWeightProvider for FIXED, COLLECT, PASSIVE
//          modes.
// Phase 8: adds INFERENCE and IMITATION modes.

#ifndef VF_ROBOT_CONTROLLER__CONTROLLER__VF_CONTROLLER_HPP_
#define VF_ROBOT_CONTROLLER__CONTROLLER__VF_CONTROLLER_HPP_

#include <memory>
#include <string>

#include <nav2_core/controller.hpp>
#include <nav2_costmap_2d/costmap_2d_ros.hpp>
#include <nav2_mppi_controller/controller.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <vf_robot_messages/msg/mppi_critics_stats.hpp>

#include "vf_robot_controller/controller/vf_critic_manager.hpp"
#include "vf_robot_controller/meta_critic/i_weight_provider.hpp"

namespace vf_robot_controller::meta_critic { class ImitationVelocityProvider; }

namespace vf_robot_controller {

enum class VFMode {
  FIXED,
  COLLECT,
  INFERENCE,   // Phase 8 — falls back to FIXED in Phase 2
  IMITATION,   // Phase 8 — falls back to FIXED in Phase 2
  PASSIVE,
};

class VFController : public nav2_core::Controller {
public:
  VFController() = default;
  ~VFController() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    const std::shared_ptr<tf2_ros::Buffer> tf,
    const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker) override;

  void setPlan(const nav_msgs::msg::Path & path) override;
  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

private:
  static VFMode parseMode(const std::string & s);
  void publishCriticDeltas(const rclcpp::Time & stamp);
  void publishAppliedWeights(const rclcpp::Time & stamp);

  rclcpp::Logger logger_{rclcpp::get_logger("VFController")};
  rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
  std::string name_;
  VFMode mode_{VFMode::FIXED};

  std::unique_ptr<nav2_mppi_controller::MPPIController> upstream_;
  std::unique_ptr<VFCriticManager> critic_manager_;
  std::shared_ptr<meta_critic::IWeightProvider> weight_provider_;

  // Phase 8: IMITATION mode reads velocity directly from a sidecar topic.
  // Owned only when mode_ == IMITATION; in all other modes it stays null.
  std::shared_ptr<meta_critic::ImitationVelocityProvider> imitation_provider_;

  // COLLECT-mode publisher: per-critic cost deltas summed across trajectories.
  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<
    vf_robot_messages::msg::MppiCriticsStats>> critic_costs_pub_;

  // COLLECT-mode publisher: per-cycle weight vector returned by the
  // WeightProvider. Consumed by data_collector_node.py on /vf/applied_weights
  // and written into critic_weights_applied of the HDF5 episode log. Without
  // this, INFERENCE training sees all-NaN targets and learns nothing.
  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<
    std_msgs::msg::Float32MultiArray>> applied_weights_pub_;

  // Empty feature vector reused across cycles (FixedWeightProvider ignores it).
  Eigen::VectorXf empty_features_;
};

}  // namespace vf_robot_controller

#endif  // VF_ROBOT_CONTROLLER__CONTROLLER__VF_CONTROLLER_HPP_
