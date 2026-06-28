// Channel: gcf_rosette, 48 dims. Implementation Phase 5.

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__FEATURES__CHANNELS__CHANNEL_GCF_ROSETTE_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__FEATURES__CHANNELS__CHANNEL_GCF_ROSETTE_HPP_

#include "vf_robot_controller/perception/features/i_feature_channel.hpp"

namespace vf_robot_controller::perception {

class GcfRosetteChannel : public IFeatureChannel {
public:
  GcfRosetteChannel();
  std::string name() const override { return "gcf_rosette"; }
  int dim() const override { return 48; }
  void compute(const PerceptionState & state, Eigen::Ref<Eigen::VectorXf> out) const override;
};

}  // namespace vf_robot_controller::perception

#endif
