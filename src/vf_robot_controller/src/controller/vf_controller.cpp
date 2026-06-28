// VFController — Phase 2 + Phase 8.
//
// Five modes layered on the Phase 1 passthrough (controller_mode strings):
//   "fixedwt"     (FIXED)     - push fixed multipliers from YAML into the WeightCache;
//                               upstream MPPI runs with Weighted* critic wrappers.
//   "collectwt"   (COLLECT)   - same as FIXED + cost-collection flag flipped + per-critic
//                               cost publisher emits MppiCriticsStats every cycle.
//   "inferencewt" (INFERENCE) - same MPPI path as FIXED, but the multipliers come from
//                               an OnnxWeightProvider that reads /vf/features and runs
//                               the trained meta-critic ONNX in-process every cycle.
//                               Falls back to FixedWeightProvider on stale features /
//                               missing onnxruntime / missing .onnx file.
//   "imitationwt" (PASSIVE)   - short-circuit, return zero twist. Python imitation
//                               sidecar owns /cmd_vel directly. Nav2 BT slot only.

#include "vf_robot_controller/controller/vf_controller.hpp"

#include <algorithm>
#include <numeric>

#include <nav2_util/node_utils.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/qos.hpp>

#include "vf_robot_controller/controller/weight_cache.hpp"
#include "vf_robot_controller/meta_critic/fixed_weight_provider.hpp"
#include "vf_robot_controller/meta_critic/imitation_velocity_provider.hpp"
#include "vf_robot_controller/meta_critic/onnx_weight_provider.hpp"
#include "vf_robot_controller/meta_critic/topic_weight_provider.hpp"

namespace vf_robot_controller {

VFMode VFController::parseMode(const std::string & s)
{
  if (s == "fixedwt")     return VFMode::FIXED;
  if (s == "collectwt")   return VFMode::COLLECT;
  if (s == "inferencewt") return VFMode::INFERENCE;
  if (s == "imitationwt") return VFMode::PASSIVE;
  return VFMode::FIXED;
}

void VFController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name,
  const std::shared_ptr<tf2_ros::Buffer> tf,
  const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  name_ = name;
  node_ = parent;
  auto node = parent.lock();
  if (node) logger_ = node->get_logger();

  // ── Mode + provider selection ──────────────────────────────────────────
  std::string mode_str = "fixedwt";
  std::string provider_str = "fixed";
  if (node) {
    nav2_util::declare_parameter_if_not_declared(
      node, name + ".controller_mode", rclcpp::ParameterValue(std::string("fixedwt")));
    nav2_util::declare_parameter_if_not_declared(
      node, name + ".weight_provider", rclcpp::ParameterValue(std::string("fixed")));
    node->get_parameter(name + ".controller_mode", mode_str);
    node->get_parameter(name + ".weight_provider", provider_str);
  }
  mode_ = parseMode(mode_str);
  RCLCPP_INFO(
    logger_, "VFController '%s' starting in mode=%s, provider=%s",
    name_.c_str(), mode_str.c_str(), provider_str.c_str());

  // ── PASSIVE mode ────────────────────────────────────────────────────────
  // Don't even configure upstream MPPI — it would chew CPU sampling
  // trajectories whose result we'd discard. Just return zero twist.
  if (mode_ == VFMode::PASSIVE) {
    WeightCache::instance().setActive(false);
    return;
  }

  // ── IMITATION mode (reserved — currently unreachable) ──────────────────
  // parseMode maps "imitationwt" -> PASSIVE (handled above), so this branch
  // is not taken by any active YAML. Kept for the future design where the
  // C++ plugin consumes the imitation sidecar's twist via
  // ImitationVelocityProvider instead of the sidecar publishing directly to
  // /cmd_vel_nav. If you wire that up, change parseMode to return IMITATION
  // for "imitationwt" and the rest of this function will work as written.
  if (mode_ == VFMode::IMITATION) {
    WeightCache::instance().setActive(false);
    imitation_provider_ = std::make_shared<meta_critic::ImitationVelocityProvider>();
    imitation_provider_->configure(parent, name);
    return;
  }

  // ── Configure upstream MPPI ─────────────────────────────────────────────
  upstream_ = std::make_unique<nav2_mppi_controller::MPPIController>();
  upstream_->configure(parent, name, tf, costmap_ros);

  // ── Read the critic list ────────────────────────────────────────────────
  // Must happen AFTER upstream->configure so the params are declared.
  std::vector<std::string> critic_short_names;
  if (node) {
    nav2_util::declare_parameter_if_not_declared(
      node, name + ".critics", rclcpp::ParameterValue(std::vector<std::string>{}));
    node->get_parameter(name + ".critics", critic_short_names);
  }

  // ── Build the critic manager + weight provider ─────────────────────────
  critic_manager_ = std::make_unique<VFCriticManager>();
  critic_manager_->configure(parent, name, critic_short_names);

  const int num_critics = static_cast<int>(critic_short_names.size());
  if (provider_str == "onnx") {
    auto onnx = std::make_shared<meta_critic::OnnxWeightProvider>();
    onnx->configure(parent, name, num_critics);
    weight_provider_ = onnx;
  } else if (provider_str == "topic") {
    auto topic = std::make_shared<meta_critic::TopicWeightProvider>();
    topic->configure(parent, name, num_critics);
    weight_provider_ = topic;
  } else if (provider_str == "fixed") {
    auto fixed = std::make_shared<meta_critic::FixedWeightProvider>();
    fixed->configure(parent, name, num_critics);
    weight_provider_ = fixed;
  } else {
    RCLCPP_WARN(
      logger_,
      "VFController: weight_provider='%s' unknown; falling back to fixed.",
      provider_str.c_str());
    auto fixed = std::make_shared<meta_critic::FixedWeightProvider>();
    fixed->configure(parent, name, num_critics);
    weight_provider_ = fixed;
  }
  critic_manager_->setWeightProvider(weight_provider_);

  // ── Publishers ─────────────────────────────────────────────────────────
  // applied_weights: available in all MPPI modes so inference can be verified
  // without switching to collect mode.
  if (node) {
    applied_weights_pub_ = node->create_publisher<std_msgs::msg::Float32MultiArray>(
      "/vf/applied_weights", rclcpp::QoS(10));
  }
  // per_critic_costs: always published so data_collector_node can record
  // from any MPPI mode (fixedwt, inferencewt) without a separate collectwt YAML.
  if (node) {
    critic_costs_pub_ = node->create_publisher<vf_robot_messages::msg::MppiCriticsStats>(
      "/vf/per_critic_costs", rclcpp::QoS(10));
  }

  // ── Activate the cache so wrappers consult it ──────────────────────────
  WeightCache::instance().setActive(true);
}

void VFController::cleanup()
{
  RCLCPP_INFO(logger_, "VFController::cleanup");
  WeightCache::instance().clear();
  if (mode_ != VFMode::PASSIVE && mode_ != VFMode::IMITATION && upstream_) {
    upstream_->cleanup();
  }
  critic_costs_pub_.reset();
  applied_weights_pub_.reset();
  critic_manager_.reset();
  weight_provider_.reset();
  imitation_provider_.reset();
}

void VFController::activate()
{
  RCLCPP_INFO(logger_, "VFController::activate");
  if (critic_costs_pub_) critic_costs_pub_->on_activate();
  if (applied_weights_pub_) applied_weights_pub_->on_activate();
  if (mode_ != VFMode::PASSIVE && mode_ != VFMode::IMITATION && upstream_) {
    upstream_->activate();
  }
}

void VFController::deactivate()
{
  RCLCPP_INFO(logger_, "VFController::deactivate");
  if (critic_costs_pub_) critic_costs_pub_->on_deactivate();
  if (applied_weights_pub_) applied_weights_pub_->on_deactivate();
  if (mode_ != VFMode::PASSIVE && mode_ != VFMode::IMITATION && upstream_) {
    upstream_->deactivate();
  }
}

geometry_msgs::msg::TwistStamped VFController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & velocity,
  nav2_core::GoalChecker * goal_checker)
{
  if (mode_ == VFMode::PASSIVE) {
    geometry_msgs::msg::TwistStamped zero;
    zero.header.stamp = pose.header.stamp;
    zero.header.frame_id = pose.header.frame_id;
    return zero;
  }

  // IMITATION: skip MPPI entirely; emit the network's twist directly.
  // Per design anti-pattern #7, this is its own runtime path — not
  // routed through the weight provider.
  if (mode_ == VFMode::IMITATION) {
    geometry_msgs::msg::TwistStamped out;
    out.header.stamp = pose.header.stamp;
    out.header.frame_id = pose.header.frame_id;
    if (imitation_provider_) {
      auto [twist, ok] = imitation_provider_->getCommand();
      out.twist = twist;
      (void)ok;  // zero on stale; velocity_smoother glides.
    }
    return out;
  }

  // Push this cycle's multipliers into the cache before delegating.
  if (critic_manager_) {
    critic_manager_->pushWeights(empty_features_);
  }

  WeightCache::instance().setCostCollectionActive(true);
  auto cmd = upstream_->computeVelocityCommands(pose, velocity, goal_checker);
  WeightCache::instance().setCostCollectionActive(false);
  publishCriticDeltas(cmd.header.stamp);
  publishAppliedWeights(cmd.header.stamp);

  return cmd;
}

void VFController::publishAppliedWeights(const rclcpp::Time & /*stamp*/)
{
  if (!applied_weights_pub_ || !weight_provider_) return;
  // FixedWeightProvider ignores features; OnnxWeightProvider reads them but
  // we pass empty for now (COLLECT mode uses FixedWeightProvider in practice).
  // The vector returned is exactly what was pushed into WeightCache this
  // cycle, so the published topic equals the actual applied multipliers.
  const auto weights = weight_provider_->getWeights(empty_features_);
  if (weights.empty()) return;
  std_msgs::msg::Float32MultiArray msg;
  msg.data.assign(weights.begin(), weights.end());
  applied_weights_pub_->publish(msg);
}

void VFController::publishCriticDeltas(const rclcpp::Time & stamp)
{
  if (!critic_costs_pub_) return;
  auto deltas = WeightCache::instance().takeRecordedDeltas();
  if (deltas.empty()) return;

  vf_robot_messages::msg::MppiCriticsStats msg;
  msg.stamp = stamp;
  msg.critics.reserve(deltas.size());
  msg.changed.reserve(deltas.size());
  msg.costs_sum.reserve(deltas.size());

  for (const auto & key : critic_manager_->criticKeys()) {
    auto it = deltas.find(key);
    msg.critics.push_back(key);
    if (it == deltas.end() || it->second.empty()) {
      msg.costs_sum.push_back(0.0f);
      msg.changed.push_back(false);
    } else {
      const float sum = std::accumulate(it->second.begin(), it->second.end(), 0.0f);
      msg.costs_sum.push_back(sum);
      msg.changed.push_back(sum != 0.0f);
    }
  }
  critic_costs_pub_->publish(msg);
}

void VFController::setPlan(const nav_msgs::msg::Path & path)
{
  if (mode_ != VFMode::PASSIVE && mode_ != VFMode::IMITATION && upstream_) {
    upstream_->setPlan(path);
  }
}

void VFController::setSpeedLimit(const double & speed_limit, const bool & percentage)
{
  if (mode_ != VFMode::PASSIVE && mode_ != VFMode::IMITATION && upstream_) {
    upstream_->setSpeedLimit(speed_limit, percentage);
  }
}

}  // namespace vf_robot_controller

PLUGINLIB_EXPORT_CLASS(vf_robot_controller::VFController, nav2_core::Controller)
