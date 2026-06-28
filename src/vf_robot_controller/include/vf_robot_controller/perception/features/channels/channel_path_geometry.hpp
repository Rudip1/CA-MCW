// Channel: path_geometry, 14 dims. Implementation Phase 5.

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__FEATURES__CHANNELS__CHANNEL_PATH_GEOMETRY_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__FEATURES__CHANNELS__CHANNEL_PATH_GEOMETRY_HPP_

#include "vf_robot_controller/perception/features/i_feature_channel.hpp"

namespace vf_robot_controller::perception {

class PathGeometryChannel : public IFeatureChannel {
public:
  PathGeometryChannel();
  std::string name() const override { return "path_geometry"; }
  int dim() const override { return 14; }
  void compute(const PerceptionState & state, Eigen::Ref<Eigen::VectorXf> out) const override;
};

}  // namespace vf_robot_controller::perception

#endif
