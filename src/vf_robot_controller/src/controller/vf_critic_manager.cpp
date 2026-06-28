#include "vf_robot_controller/controller/vf_critic_manager.hpp"

#include <utility>

#include "vf_robot_controller/controller/weight_cache.hpp"

namespace vf_robot_controller {

void VFCriticManager::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & node,
  const std::string & parent_name,
  const std::vector<std::string> & critic_short_names)
{
  if (auto n = node.lock()) {
    logger_ = n->get_logger();
  }
  critic_short_names_ = critic_short_names;
  critic_keys_.clear();
  critic_keys_.reserve(critic_short_names.size());
  for (const auto & short_name : critic_short_names) {
    critic_keys_.push_back(parent_name + "." + short_name);
  }
  RCLCPP_INFO(
    logger_, "VFCriticManager: tracking %zu critics under parent '%s'",
    critic_keys_.size(), parent_name.c_str());
}

void VFCriticManager::setWeightProvider(
  std::shared_ptr<meta_critic::IWeightProvider> provider)
{
  provider_ = std::move(provider);
}

void VFCriticManager::pushWeights(const Eigen::Ref<const Eigen::VectorXf> & features)
{
  if (!provider_) return;

  auto weights = provider_->getWeights(features);
  auto & cache = WeightCache::instance();

  if (weights.empty()) {
    // Provider has no override this cycle — fall back to multiplier == 1.0.
    for (const auto & key : critic_keys_) {
      cache.setMultiplier(key, 1.0f);
    }
    return;
  }

  if (weights.size() != critic_keys_.size()) {
    RCLCPP_WARN_THROTTLE(
      logger_, *rclcpp::Clock::make_shared(), 5000,
      "VFCriticManager: provider returned %zu weights, expected %zu. Padding.",
      weights.size(), critic_keys_.size());
  }

  for (size_t i = 0; i < critic_keys_.size(); ++i) {
    const float w = (i < weights.size()) ? weights[i] : 1.0f;
    cache.setMultiplier(critic_keys_[i], w);
  }
}

void VFCriticManager::setCostCollectionActive(bool active)
{
  WeightCache::instance().setCostCollectionActive(active);
}

std::unordered_map<std::string, std::vector<float>>
VFCriticManager::takeRecordedDeltas()
{
  return WeightCache::instance().takeRecordedDeltas();
}

}  // namespace vf_robot_controller
