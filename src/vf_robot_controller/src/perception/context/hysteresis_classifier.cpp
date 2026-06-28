#include "vf_robot_controller/perception/context/hysteresis_classifier.hpp"

namespace vf_robot_controller::perception {

NavigationContext HysteresisClassifier::classify(const PerceptionState & state)
{
  // APPROACHING_GOAL overrides everything when very close to the goal.
  if (state.distance_to_goal > 0.0f &&
      state.distance_to_goal < thresholds_.approach_distance)
  {
    last_ = NavigationContext::APPROACHING_GOAL;
    return last_;
  }

  const float g = state.gcf_scalar;
  const auto & t = thresholds_;

  switch (last_) {
    case NavigationContext::OPEN:
      if (g > t.open_low) {
        last_ = (g > t.tight_high) ? NavigationContext::CORRIDOR
                                   : NavigationContext::CLUTTERED;
      }
      break;
    case NavigationContext::CORRIDOR:
      if (g < t.tight_low) last_ = NavigationContext::CLUTTERED;
      if (g < t.open_high) last_ = NavigationContext::OPEN;
      if (g > t.clutter_dynamic) last_ = NavigationContext::DYNAMIC;
      break;
    case NavigationContext::CLUTTERED:
      if (g > t.tight_high) last_ = NavigationContext::CORRIDOR;
      if (g < t.open_high)  last_ = NavigationContext::OPEN;
      break;
    case NavigationContext::DYNAMIC:
      if (g < t.tight_low) last_ = NavigationContext::CORRIDOR;
      break;
    case NavigationContext::DOORWAY:
    case NavigationContext::APPROACHING_GOAL:
    case NavigationContext::UNKNOWN:
    default:
      // Fresh classification.
      if (g > t.clutter_dynamic) last_ = NavigationContext::DYNAMIC;
      else if (g > t.tight_high) last_ = NavigationContext::CORRIDOR;
      else if (g > t.open_low)   last_ = NavigationContext::CLUTTERED;
      else                        last_ = NavigationContext::OPEN;
      break;
  }

  return last_;
}

}  // namespace vf_robot_controller::perception
