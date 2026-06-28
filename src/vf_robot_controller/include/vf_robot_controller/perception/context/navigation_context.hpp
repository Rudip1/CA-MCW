// NavigationContext enum and helpers. Locked in Phase 0.

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__CONTEXT__NAVIGATION_CONTEXT_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__CONTEXT__NAVIGATION_CONTEXT_HPP_

#include <cstdint>
#include <string>

namespace vf_robot_controller::perception {

enum class NavigationContext : uint8_t {
  OPEN              = 0,
  CORRIDOR          = 1,
  DOORWAY           = 2,
  DYNAMIC           = 3,
  CLUTTERED         = 4,
  APPROACHING_GOAL  = 5,
  UNKNOWN           = 255,
};

inline std::string contextName(NavigationContext c) {
  switch (c) {
    case NavigationContext::OPEN:             return "open";
    case NavigationContext::CORRIDOR:         return "corridor";
    case NavigationContext::DOORWAY:          return "doorway";
    case NavigationContext::DYNAMIC:          return "dynamic";
    case NavigationContext::CLUTTERED:        return "cluttered";
    case NavigationContext::APPROACHING_GOAL: return "approaching_goal";
    default: return "unknown";
  }
}

}  // namespace vf_robot_controller::perception

#endif
