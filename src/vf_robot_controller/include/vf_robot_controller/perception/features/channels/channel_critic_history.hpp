// Channel: critic_history, 30 dims. Implementation Phase 5.

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__FEATURES__CHANNELS__CHANNEL_CRITIC_HISTORY_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__FEATURES__CHANNELS__CHANNEL_CRITIC_HISTORY_HPP_

#include "vf_robot_controller/perception/features/i_feature_channel.hpp"

namespace vf_robot_controller::perception {

class CriticHistoryChannel : public IFeatureChannel {
public:
  CriticHistoryChannel();
  std::string name() const override { return "critic_history"; }
  int dim() const override { return 30; }
  void compute(const PerceptionState & state, Eigen::Ref<Eigen::VectorXf> out) const override;
};

}  // namespace vf_robot_controller::perception

#endif
