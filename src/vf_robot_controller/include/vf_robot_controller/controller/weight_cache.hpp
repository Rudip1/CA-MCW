// WeightCache — process-wide cache of per-critic meta-weight multipliers and
// per-critic cost deltas. Phase 2.
//
// Why a process-wide singleton:
//   The upstream Optimizer holds CriticManager by value, and its
//   evalTrajectoriesScores is non-virtual. We can't inject state through the
//   Optimizer or CriticManager. But each critic plugin instance is created
//   inside the optimizer at configure time and lives for the controller's
//   lifetime. The cleanest channel between VFController and its wrapped
//   critic plugins is shared mutable state keyed by the critic's full name.
//
// Header-only (C++17 inline static) so libvf_controller_lib.so and
// libvf_critics_lib.so resolve to the same instance via the dynamic linker
// without forming an explicit link-time dependency between them.

#ifndef VF_ROBOT_CONTROLLER__CONTROLLER__WEIGHT_CACHE_HPP_
#define VF_ROBOT_CONTROLLER__CONTROLLER__WEIGHT_CACHE_HPP_

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vf_robot_controller {

class WeightCache {
public:
  static WeightCache & instance()
  {
    static WeightCache cache;
    return cache;
  }

  void setMultiplier(const std::string & critic_name, float multiplier)
  {
    std::lock_guard<std::mutex> lock(mu_);
    multipliers_[critic_name] = multiplier;
  }

  std::optional<float> getMultiplier(const std::string & critic_name) const
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = multipliers_.find(critic_name);
    if (it == multipliers_.end()) return std::nullopt;
    return it->second;
  }

  void setActive(bool active)
  {
    std::lock_guard<std::mutex> lock(mu_);
    active_ = active;
  }

  bool isActive() const
  {
    std::lock_guard<std::mutex> lock(mu_);
    return active_;
  }

  void setCostCollectionActive(bool active)
  {
    std::lock_guard<std::mutex> lock(mu_);
    cost_collection_active_ = active;
    if (active) deltas_.clear();
  }

  bool isCostCollectionActive() const
  {
    std::lock_guard<std::mutex> lock(mu_);
    return cost_collection_active_;
  }

  void recordDelta(const std::string & critic_name, std::vector<float> delta)
  {
    std::lock_guard<std::mutex> lock(mu_);
    deltas_[critic_name] = std::move(delta);
  }

  std::unordered_map<std::string, std::vector<float>> takeRecordedDeltas()
  {
    std::lock_guard<std::mutex> lock(mu_);
    std::unordered_map<std::string, std::vector<float>> out;
    out.swap(deltas_);
    return out;
  }

  void clear()
  {
    std::lock_guard<std::mutex> lock(mu_);
    multipliers_.clear();
    deltas_.clear();
    active_ = false;
    cost_collection_active_ = false;
  }

private:
  WeightCache() = default;

  mutable std::mutex mu_;
  std::unordered_map<std::string, float> multipliers_;
  std::unordered_map<std::string, std::vector<float>> deltas_;
  bool active_{false};
  bool cost_collection_active_{false};
};

}  // namespace vf_robot_controller

#endif  // VF_ROBOT_CONTROLLER__CONTROLLER__WEIGHT_CACHE_HPP_
