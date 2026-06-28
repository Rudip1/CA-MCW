// Channel: obstacle_dynamics, 16 dims. Implementation Phase 5.

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__FEATURES__CHANNELS__CHANNEL_OBSTACLE_DYNAMICS_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__FEATURES__CHANNELS__CHANNEL_OBSTACLE_DYNAMICS_HPP_

#include "vf_robot_controller/perception/features/i_feature_channel.hpp"

namespace vf_robot_controller::perception {

class ObstacleDynamicsChannel : public IFeatureChannel {
public:
  ObstacleDynamicsChannel();
  std::string name() const override { return "obstacle_dynamics"; }
  int dim() const override { return 16; }
  void compute(const PerceptionState & state, Eigen::Ref<Eigen::VectorXf> out) const override;
};

}  // namespace vf_robot_controller::perception

#endif
