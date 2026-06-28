// IFeatureChannel — interface every feature channel implements.
// FeatureExtractor concatenates channels in YAML-declared order.

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__FEATURES__I_FEATURE_CHANNEL_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__FEATURES__I_FEATURE_CHANNEL_HPP_

#include <string>
#include <Eigen/Core>
#include "vf_robot_controller/perception/common/types.hpp"

namespace vf_robot_controller::perception {

class IFeatureChannel {
public:
  virtual ~IFeatureChannel() = default;

  /// Channel name (matches YAML config key).
  virtual std::string name() const = 0;

  /// Output dimensionality (declared at construction).
  virtual int dim() const = 0;

  /// Compute channel features into out (already sized to dim()).
  virtual void compute(const PerceptionState & state, Eigen::Ref<Eigen::VectorXf> out) const = 0;
};

}  // namespace vf_robot_controller::perception

#endif
