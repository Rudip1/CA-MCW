// FeatureExtractor — orchestrates an ordered list of IFeatureChannels.
// Phase 5.

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__FEATURES__FEATURE_EXTRACTOR_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__FEATURES__FEATURE_EXTRACTOR_HPP_

#include <memory>
#include <string>
#include <vector>
#include <Eigen/Core>
#include "vf_robot_controller/perception/features/i_feature_channel.hpp"

namespace vf_robot_controller::perception {

class FeatureExtractor {
public:
  FeatureExtractor() = default;
  void addChannel(std::unique_ptr<IFeatureChannel> channel);
  int totalDim() const;
  Eigen::VectorXf extract(const PerceptionState & state) const;

  // Channel introspection (for /vf/features payload metadata)
  std::vector<std::string> channelNames() const;
  std::vector<int> channelDims() const;

private:
  std::vector<std::unique_ptr<IFeatureChannel>> channels_;
};

// Factory: name → IFeatureChannel. Returns nullptr for unknown names.
// The factory keeps Phase 5 channels here; new channels register at the
// bottom of channel_factory.cpp.
std::unique_ptr<IFeatureChannel> makeChannel(const std::string & name);

}  // namespace vf_robot_controller::perception

#endif
