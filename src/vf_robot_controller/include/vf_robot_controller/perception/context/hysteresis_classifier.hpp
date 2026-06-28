// HysteresisClassifier — geometric context classifier with hysteresis. Phase 5.
//
// Rules of thumb (cheap, GCF-driven):
//   gcf > tight_high      → CORRIDOR / DOORWAY (depending on velocity)
//   gcf > medium          → CLUTTERED
//   gcf < open_low        → OPEN
//   distance_to_goal small → APPROACHING_GOAL (overrides geometry)
//
// Hysteresis: once we transition into a state, we don't leave until the
// signal crosses the *opposite* threshold. Avoids per-cycle flapping when
// gcf hovers near a boundary.

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__CONTEXT__HYSTERESIS_CLASSIFIER_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__CONTEXT__HYSTERESIS_CLASSIFIER_HPP_

#include "vf_robot_controller/perception/context/i_context_classifier.hpp"

namespace vf_robot_controller::perception {

struct HysteresisThresholds {
  float open_high{0.20f};       // gcf below this → safely OPEN
  float open_low{0.30f};        // gcf above this exits OPEN
  float tight_low{0.55f};       // gcf below this exits CORRIDOR
  float tight_high{0.65f};      // gcf above this enters CORRIDOR
  float clutter_dynamic{0.85f}; // very high gcf → CLUTTERED
  float approach_distance{1.0f};// metres to goal that triggers APPROACHING_GOAL
};

class HysteresisClassifier : public IContextClassifier {
public:
  HysteresisClassifier() = default;
  explicit HysteresisClassifier(HysteresisThresholds t) : thresholds_(t) {}

  void setThresholds(const HysteresisThresholds & t) { thresholds_ = t; }
  NavigationContext classify(const PerceptionState & state) override;

private:
  HysteresisThresholds thresholds_;
  NavigationContext last_{NavigationContext::UNKNOWN};
};

}  // namespace vf_robot_controller::perception

#endif
