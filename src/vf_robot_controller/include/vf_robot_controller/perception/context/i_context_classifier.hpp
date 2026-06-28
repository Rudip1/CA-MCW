// IContextClassifier — interface for classifying current navigation context.
// Phase 5 implementation.

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__CONTEXT__I_CONTEXT_CLASSIFIER_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__CONTEXT__I_CONTEXT_CLASSIFIER_HPP_

#include "vf_robot_controller/perception/context/navigation_context.hpp"
#include "vf_robot_controller/perception/common/types.hpp"

namespace vf_robot_controller::perception {

class IContextClassifier {
public:
  virtual ~IContextClassifier() = default;
  virtual NavigationContext classify(const PerceptionState & state) = 0;
};

}  // namespace vf_robot_controller::perception

#endif
