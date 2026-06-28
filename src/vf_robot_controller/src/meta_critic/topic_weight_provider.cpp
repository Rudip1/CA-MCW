// TopicWeightProvider — Phase 8.

#include "vf_robot_controller/meta_critic/topic_weight_provider.hpp"

#include <algorithm>

#include <nav2_util/node_utils.hpp>
#include <rclcpp/qos.hpp>

namespace vf_robot_controller::meta_critic {

TopicWeightProvider::TopicWeightProvider() = default;

void TopicWeightProvider::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & node_weak,
  const std::string & param_ns,
  int num_critics)
{
  num_critics_ = num_critics;
  node_weak_ = node_weak;
  fallback_weights_.assign(num_critics, 1.0f);

  auto node = node_weak.lock();
  if (!node) {
    return;
  }
  logger_ = node->get_logger();

  // ── Parameters ───────────────────────────────────────────────────────────
  std::string topic = "/vf_controller/meta_weights";
  int timeout_ms = 200;
  std::vector<double> fallback;
  nav2_util::declare_parameter_if_not_declared(
    node, param_ns + ".weight_topic", rclcpp::ParameterValue(topic));
  nav2_util::declare_parameter_if_not_declared(
    node, param_ns + ".weight_timeout_ms", rclcpp::ParameterValue(timeout_ms));
  nav2_util::declare_parameter_if_not_declared(
    node, param_ns + ".fixed_weights",
    rclcpp::ParameterValue(std::vector<double>{}));
  node->get_parameter(param_ns + ".weight_topic", topic);
  node->get_parameter(param_ns + ".weight_timeout_ms", timeout_ms);
  node->get_parameter(param_ns + ".fixed_weights", fallback);
  timeout_ = std::chrono::milliseconds(std::max(50, timeout_ms));
  if (!fallback.empty()) {
    const int n = std::min<int>(
      num_critics_, static_cast<int>(fallback.size()));
    for (int i = 0; i < n; ++i) {
      fallback_weights_[i] = static_cast<float>(fallback[i]);
    }
  }

  // ── Subscriber ───────────────────────────────────────────────────────────
  rclcpp::QoS qos(rclcpp::KeepLast(10));
  qos.best_effort();
  sub_ = node->create_subscription<std_msgs::msg::Float32MultiArray>(
    topic, qos,
    [this](std_msgs::msg::Float32MultiArray::SharedPtr m) { onWeightsMsg(m); });

  RCLCPP_INFO(
    logger_,
    "TopicWeightProvider: subscribed to '%s' (timeout=%d ms, K=%d, fallback=fixed_weights)",
    topic.c_str(), timeout_ms, num_critics_);
}

void TopicWeightProvider::onWeightsMsg(
  const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  std::lock_guard<std::mutex> g(mu_);
  latest_weights_.assign(msg->data.begin(), msg->data.end());
  if (auto node = node_weak_.lock()) {
    latest_stamp_ = node->now();
  }
  warned_stale_once_ = false;
}

int TopicWeightProvider::numCritics() const { return num_critics_; }

std::vector<float> TopicWeightProvider::getWeights(
  const Eigen::Ref<const Eigen::VectorXf> & /*features*/)
{
  std::lock_guard<std::mutex> g(mu_);
  auto node = node_weak_.lock();
  if (!node || latest_weights_.empty()) {
    if (!warned_stale_once_) {
      RCLCPP_WARN(logger_, "TopicWeightProvider: no weights yet; using fallback.");
      warned_stale_once_ = true;
    }
    return fallback_weights_;
  }
  const auto age = node->now() - latest_stamp_;
  if (age > rclcpp::Duration(timeout_)) {
    if (!warned_stale_once_) {
      RCLCPP_WARN(
        logger_, "TopicWeightProvider: stale (age=%.0f ms); using fallback.",
        age.seconds() * 1000.0);
      warned_stale_once_ = true;
    }
    return fallback_weights_;
  }

  std::vector<float> out(num_critics_, 1.0f);
  const int n = std::min<int>(num_critics_, static_cast<int>(latest_weights_.size()));
  for (int i = 0; i < n; ++i) {
    out[i] = latest_weights_[i];
  }
  return out;
}

void TopicWeightProvider::injectMessageForTest(const std::vector<float> & weights)
{
  std::lock_guard<std::mutex> g(mu_);
  latest_weights_ = weights;
  if (auto node = node_weak_.lock()) {
    latest_stamp_ = node->now();
  }
}

}  // namespace vf_robot_controller::meta_critic
