// Channel: reynolds, 4 dims. Implementation Phase 5.

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__FEATURES__CHANNELS__CHANNEL_REYNOLDS_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__FEATURES__CHANNELS__CHANNEL_REYNOLDS_HPP_

#include "vf_robot_controller/perception/features/i_feature_channel.hpp"

namespace vf_robot_controller::perception {

class ReynoldsChannel : public IFeatureChannel {
public:
  ReynoldsChannel();
  std::string name() const override { return "reynolds"; }
  int dim() const override { return 4; }
  void compute(const PerceptionState & state, Eigen::Ref<Eigen::VectorXf> out) const override;
};

}  // namespace vf_robot_controller::perception

#endif
