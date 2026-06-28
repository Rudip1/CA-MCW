// FeatureExtractor implementation + channel factory. Phase 5.

#include "vf_robot_controller/perception/features/feature_extractor.hpp"

#include "vf_robot_controller/perception/features/channels/channel_context.hpp"
#include "vf_robot_controller/perception/features/channels/channel_critic_history.hpp"
#include "vf_robot_controller/perception/features/channels/channel_gcf_rosette.hpp"
#include "vf_robot_controller/perception/features/channels/channel_obstacle_dynamics.hpp"
#include "vf_robot_controller/perception/features/channels/channel_path_geometry.hpp"
#include "vf_robot_controller/perception/features/channels/channel_reynolds.hpp"
#include "vf_robot_controller/perception/features/channels/channel_robot_state.hpp"
#include "vf_robot_controller/perception/features/channels/channel_slam_persistent.hpp"

namespace vf_robot_controller::perception {

void FeatureExtractor::addChannel(std::unique_ptr<IFeatureChannel> channel)
{
  channels_.push_back(std::move(channel));
}

int FeatureExtractor::totalDim() const
{
  int total = 0;
  for (const auto & c : channels_) total += c->dim();
  return total;
}

Eigen::VectorXf FeatureExtractor::extract(const PerceptionState & state) const
{
  Eigen::VectorXf out(totalDim());
  int offset = 0;
  for (const auto & c : channels_) {
    c->compute(state, out.segment(offset, c->dim()));
    offset += c->dim();
  }
  return out;
}

std::vector<std::string> FeatureExtractor::channelNames() const
{
  std::vector<std::string> names;
  names.reserve(channels_.size());
  for (const auto & c : channels_) names.push_back(c->name());
  return names;
}

std::vector<int> FeatureExtractor::channelDims() const
{
  std::vector<int> dims;
  dims.reserve(channels_.size());
  for (const auto & c : channels_) dims.push_back(c->dim());
  return dims;
}

std::unique_ptr<IFeatureChannel> makeChannel(const std::string & name)
{
  if (name == "robot_state")        return std::make_unique<RobotStateChannel>();
  if (name == "context")            return std::make_unique<ContextChannel>();
  if (name == "path_geometry")      return std::make_unique<PathGeometryChannel>();
  if (name == "gcf_rosette")        return std::make_unique<GcfRosetteChannel>();
  if (name == "critic_history")     return std::make_unique<CriticHistoryChannel>();
  if (name == "obstacle_dynamics")  return std::make_unique<ObstacleDynamicsChannel>();
  if (name == "reynolds")           return std::make_unique<ReynoldsChannel>();
  if (name == "slam_persistent")    return std::make_unique<SlamPersistentChannel>();
  return nullptr;
}

}  // namespace vf_robot_controller::perception
