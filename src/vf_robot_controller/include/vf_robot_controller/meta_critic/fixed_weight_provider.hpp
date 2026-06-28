// FixedWeightProvider — Phase 2 implementation.
//
// Loads a static set of per-critic multipliers from YAML at configure() time
// and returns them every cycle, regardless of features. Used by FIXED mode
// for the thesis baseline (equal weights ≈ stock MPPI) and by emphasis-mode
// sanity checks (skewed weights → visibly different behavior).

#ifndef VF_ROBOT_CONTROLLER__META_CRITIC__FIXED_WEIGHT_PROVIDER_HPP_
#define VF_ROBOT_CONTROLLER__META_CRITIC__FIXED_WEIGHT_PROVIDER_HPP_

#include <string>
#include <vector>

#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include "vf_robot_controller/meta_critic/i_weight_provider.hpp"

namespace vf_robot_controller::meta_critic {

class FixedWeightProvider : public IWeightProvider {
public:
  FixedWeightProvider() = default;

  // Reads `<param_ns>.fixed_weights` (vector<double>) from the node and
  // stores it as the constant multiplier vector. If the parameter is absent
  // or empty, the provider is configured with `num_critics` ones.
  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & node,
    const std::string & param_ns,
    int num_critics);

  int numCritics() const override;
  std::vector<float> getWeights(
    const Eigen::Ref<const Eigen::VectorXf> & features) override;
  std::string name() const override { return "fixed"; }

  // Test-only direct setter; bypasses YAML.
  void setWeightsForTest(std::vector<float> weights);

private:
  std::vector<float> weights_;
};

}  // namespace vf_robot_controller::meta_critic

#endif  // VF_ROBOT_CONTROLLER__META_CRITIC__FIXED_WEIGHT_PROVIDER_HPP_
