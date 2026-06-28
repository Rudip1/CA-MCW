#include "vf_robot_controller/meta_critic/fixed_weight_provider.hpp"

#include <nav2_util/node_utils.hpp>
#include <rclcpp/logging.hpp>

namespace vf_robot_controller::meta_critic {

void FixedWeightProvider::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & node_weak,
  const std::string & param_ns,
  int num_critics)
{
  auto node = node_weak.lock();
  if (!node) {
    weights_.assign(num_critics, 1.0f);
    return;
  }

  const std::string key = param_ns + ".fixed_weights";
  std::vector<double> raw;
  nav2_util::declare_parameter_if_not_declared(
    node, key, rclcpp::ParameterValue(std::vector<double>{}));
  node->get_parameter(key, raw);

  if (raw.empty()) {
    weights_.assign(num_critics, 1.0f);
    RCLCPP_INFO(
      node->get_logger(),
      "FixedWeightProvider: no '%s' set, defaulting to %d ones",
      key.c_str(), num_critics);
    return;
  }

  if (static_cast<int>(raw.size()) != num_critics) {
    RCLCPP_WARN(
      node->get_logger(),
      "FixedWeightProvider: '%s' length %zu does not match critic count %d. "
      "Truncating or padding with 1.0.",
      key.c_str(), raw.size(), num_critics);
  }

  weights_.assign(num_critics, 1.0f);
  const int n = std::min<int>(num_critics, static_cast<int>(raw.size()));
  for (int i = 0; i < n; ++i) {
    weights_[i] = static_cast<float>(raw[i]);
  }
}

int FixedWeightProvider::numCritics() const
{
  return static_cast<int>(weights_.size());
}

std::vector<float> FixedWeightProvider::getWeights(
  const Eigen::Ref<const Eigen::VectorXf> & /*features*/)
{
  return weights_;
}

void FixedWeightProvider::setWeightsForTest(std::vector<float> weights)
{
  weights_ = std::move(weights);
}

}  // namespace vf_robot_controller::meta_critic
