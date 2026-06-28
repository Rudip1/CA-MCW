// WeightedCriticWrapper — Phase 2 weight-injection mechanism.
//
// Each wrapper inherits from one specific upstream critic class. Pluginlib
// instantiates the wrapper instead of the upstream critic; YAML lists wrapper
// names like "WeightedPathFollowCritic" in the FollowPath.critics array.
//
// The wrapper does two things in score():
//   1. Looks up its meta-multiplier in the WeightCache (keyed by full name).
//      If present, sets weight_ = yaml_weight × multiplier. With multiplier
//      == 1.0 (or no entry), behavior is identical to the unwrapped upstream
//      critic.
//   2. If cost collection is active, snapshots data.costs before delegating,
//      computes the delta after upstream::score, and pushes it into the
//      cache for VFController to publish on /vf/per_critic_costs.
//
// Why inheritance and not pluginlib delegation:
//   weight_ is protected on every upstream critic (path_follow_critic.hpp:55,
//   etc.). Subclassing makes it accessible without any casting tricks.
//   Pluginlib delegation would need a separate ClassLoader for the inner
//   critic and would cost double parameter declaration.
//
// Why namespace mppi::critics (not vf_robot_controller::critics):
//   Upstream's CriticManager::getFullName() (nav2_mppi_controller's
//   critic_manager.cpp:72) hardcodes the prefix "mppi::critics::" when
//   resolving YAML `critics:` short names. Anything registered in another
//   namespace will not be found by upstream's plugin loader. Putting our
//   wrappers in mppi::critics is the standard ROS plugin-author pattern
//   when extending a system that hardcodes its lookup prefix.
//
// ObstaclesCritic is intentionally NOT wrapped here — it has a dual weight
// (repulsion_weight_, critical_weight_) that needs a specialization. Phase 3
// will add WeightedObstaclesCritic if ObstaclesCritic is used in the final
// critic set; for Phase 2 it can stay unwrapped (its weights remain pure
// YAML, untouched by the meta-critic).

#ifndef VF_ROBOT_CONTROLLER__CRITICS__WEIGHTED_CRITIC_WRAPPER_HPP_
#define VF_ROBOT_CONTROLLER__CRITICS__WEIGHTED_CRITIC_WRAPPER_HPP_

#include <vector>

#include <nav2_mppi_controller/critics/constraint_critic.hpp>
#include <nav2_mppi_controller/critics/cost_critic.hpp>
#include <nav2_mppi_controller/critics/goal_angle_critic.hpp>
#include <nav2_mppi_controller/critics/goal_critic.hpp>
#include <nav2_mppi_controller/critics/path_align_critic.hpp>
#include <nav2_mppi_controller/critics/path_angle_critic.hpp>
#include <nav2_mppi_controller/critics/path_follow_critic.hpp>
#include <nav2_mppi_controller/critics/prefer_forward_critic.hpp>
#include <nav2_mppi_controller/critics/twirling_critic.hpp>
#include <nav2_mppi_controller/critics/velocity_deadband_critic.hpp>

#include "vf_robot_controller/controller/weight_cache.hpp"

namespace mppi::critics {

// Generic wrapper that captures the YAML-set weight at initialize() time and
// applies a meta-multiplier from the WeightCache before each score() call.
template <typename UpstreamT>
class WeightedCriticBase : public UpstreamT {
public:
  void initialize() override
  {
    UpstreamT::initialize();
    // Snapshot the weight set by upstream's parameter read so that
    // multiplier == 1.0 reproduces stock behavior exactly.
    yaml_weight_ = this->weight_;
  }

  void score(mppi::CriticData & data) override
  {
    auto & cache = ::vf_robot_controller::WeightCache::instance();
    if (cache.isActive()) {
      auto m = cache.getMultiplier(this->getName());
      this->weight_ = yaml_weight_ * (m.has_value() ? *m : 1.0f);
    }

    const bool collect = cache.isCostCollectionActive();
    std::vector<float> before;
    if (collect) {
      before.assign(data.costs.cbegin(), data.costs.cend());
    }

    UpstreamT::score(data);

    if (collect) {
      std::vector<float> delta(data.costs.size());
      auto after_it = data.costs.cbegin();
      for (size_t i = 0; i < delta.size(); ++i, ++after_it) {
        delta[i] = *after_it - before[i];
      }
      cache.recordDelta(this->getName(), std::move(delta));
    }
  }

protected:
  float yaml_weight_{1.0f};
};

// Concrete subclasses — one per upstream critic. Each is registered with
// pluginlib in plugins/critic_plugins.xml and listed by short name in the
// controller YAML's critics array.
class WeightedConstraintCritic : public WeightedCriticBase<mppi::critics::ConstraintCritic> {};
class WeightedCostCritic       : public WeightedCriticBase<mppi::critics::CostCritic> {};
class WeightedGoalCritic       : public WeightedCriticBase<mppi::critics::GoalCritic> {};
class WeightedGoalAngleCritic  : public WeightedCriticBase<mppi::critics::GoalAngleCritic> {};
class WeightedPathAlignCritic  : public WeightedCriticBase<mppi::critics::PathAlignCritic> {};
class WeightedPathAngleCritic  : public WeightedCriticBase<mppi::critics::PathAngleCritic> {};
class WeightedPathFollowCritic : public WeightedCriticBase<mppi::critics::PathFollowCritic> {};
class WeightedPreferForwardCritic : public WeightedCriticBase<mppi::critics::PreferForwardCritic> {};
class WeightedTwirlingCritic   : public WeightedCriticBase<mppi::critics::TwirlingCritic> {};
class WeightedVelocityDeadbandCritic : public WeightedCriticBase<mppi::critics::VelocityDeadbandCritic> {};

}  // namespace vf_robot_controller::critics

#endif  // VF_ROBOT_CONTROLLER__CRITICS__WEIGHTED_CRITIC_WRAPPER_HPP_
