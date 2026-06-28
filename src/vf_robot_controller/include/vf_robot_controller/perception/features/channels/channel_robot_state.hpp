// Channel: robot_state, 9 dims. Implementation Phase 5.

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__FEATURES__CHANNELS__CHANNEL_ROBOT_STATE_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__FEATURES__CHANNELS__CHANNEL_ROBOT_STATE_HPP_

#include "vf_robot_controller/perception/features/i_feature_channel.hpp"

namespace vf_robot_controller::perception {

class RobotStateChannel : public IFeatureChannel {
public:
  RobotStateChannel();
  std::string name() const override { return "robot_state"; }
  int dim() const override { return 9; }
  void compute(const PerceptionState & state, Eigen::Ref<Eigen::VectorXf> out) const override;
};

}  // namespace vf_robot_controller::perception

#endif
