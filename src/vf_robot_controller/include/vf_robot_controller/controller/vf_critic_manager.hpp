// VFCriticManager — sidecar that pushes per-critic multipliers from an
// IWeightProvider into the WeightCache before each control cycle.
//
// Not a subclass of upstream's CriticManager — that's a value member of
// Optimizer with non-virtual evalTrajectoriesScores, so subclassing yields
// no override hook. Instead, weight injection happens via the WeightCache:
// VFCriticManager writes multipliers indexed by full critic name, and the
// pluginlib-loaded Weighted* critic wrappers read them inside score().
//
// Phase 2.

#ifndef VF_ROBOT_CONTROLLER__CONTROLLER__VF_CRITIC_MANAGER_HPP_
#define VF_ROBOT_CONTROLLER__CONTROLLER__VF_CRITIC_MANAGER_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include "vf_robot_controller/meta_critic/i_weight_provider.hpp"

namespace vf_robot_controller {

class VFCriticManager {
public:
  VFCriticManager() = default;

  // Configure the manager with the controller's parent name (e.g. "FollowPath")
  // and the ordered list of critic short names from YAML.
  // Cache keys are constructed as "<parent_name>.<short_name>" — this matches
  // what upstream's CriticManager passes to each critic's on_configure() and
  // therefore what each Weighted* wrapper sees as its getName().
  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & node,
    const std::string & parent_name,
    const std::vector<std::string> & critic_short_names);

  void setWeightProvider(std::shared_ptr<meta_critic::IWeightProvider> provider);

  // Pull weights from provider, write them into the WeightCache. Called
  // once per controller cycle right before delegating to upstream MPPI.
  // `features` is forwarded to the provider; FixedWeightProvider ignores it.
  void pushWeights(const Eigen::Ref<const Eigen::VectorXf> & features);

  // Flip cost-collection in the WeightCache. When true, wrappers snapshot
  // data.costs around their inner score() call and record per-critic deltas
  // for the controller to publish.
  void setCostCollectionActive(bool active);

  // Pull and clear the deltas accumulated during the last cycle.
  std::unordered_map<std::string, std::vector<float>> takeRecordedDeltas();

  const std::vector<std::string> & criticKeys() const { return critic_keys_; }
  int numCritics() const { return static_cast<int>(critic_keys_.size()); }

private:
  std::vector<std::string> critic_keys_;  // FQNs: "FollowPath.WeightedPathFollowCritic"
  std::vector<std::string> critic_short_names_;
  std::shared_ptr<meta_critic::IWeightProvider> provider_;
  rclcpp::Logger logger_{rclcpp::get_logger("VFCriticManager")};
};

}  // namespace vf_robot_controller

#endif  // VF_ROBOT_CONTROLLER__CONTROLLER__VF_CRITIC_MANAGER_HPP_
