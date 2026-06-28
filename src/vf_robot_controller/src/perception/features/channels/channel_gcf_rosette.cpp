// GcfRosetteChannel — 48 dims = 16 angles × 3 fields. Phase 5.
//
// Polar samples around the robot pose at 16 evenly-spaced angles, each
// returning (composite, clearance_2d, clutter_density). The query points
// sit at a fixed `sample_radius` (default 1.0 m) ahead of the pose along
// each direction.
//
// If `state.gcf_composite` is set we run real per-angle queries.
// Otherwise we fall back to the at-pose scalar (state.gcf_scalar) on the
// composite channel and zero the clearance/clutter fields. This lets the
// channel produce a usable signal even before gcf_node is publishing the
// composite as a shared resource.

#include "vf_robot_controller/perception/features/channels/channel_gcf_rosette.hpp"

#include <cmath>

#include "vf_robot_controller/perception/gcf/gcf_composite.hpp"

namespace vf_robot_controller::perception {

namespace {
constexpr int kAngles = 16;
constexpr float kSampleRadius = 1.0f;  // metres
constexpr float kTwoPi = 6.28318530718f;
}  // namespace

GcfRosetteChannel::GcfRosetteChannel() = default;

void GcfRosetteChannel::compute(
  const PerceptionState & state, Eigen::Ref<Eigen::VectorXf> out) const
{
  out.setZero();

  if (state.gcf_composite) {
    const float rx = static_cast<float>(state.robot_pose.x);
    const float ry = static_cast<float>(state.robot_pose.y);
    const float rt = static_cast<float>(state.robot_pose.theta);
    for (int i = 0; i < kAngles; ++i) {
      const float angle = rt + (kTwoPi * i) / kAngles;
      const double qx = rx + kSampleRadius * std::cos(angle);
      const double qy = ry + kSampleRadius * std::sin(angle);
      const auto cell = state.gcf_composite->query(qx, qy);
      out(i * 3 + 0) = static_cast<float>(cell.complexity);
      out(i * 3 + 1) = static_cast<float>(cell.clearance_2d);
      out(i * 3 + 2) = static_cast<float>(cell.clutter_density);
    }
    return;
  }

  // Fallback: broadcast the at-pose scalar. Indicates "we have a value but
  // can't sample at angles". The downstream MLP should still find this
  // useful, just with reduced spatial resolution.
  if (state.gcf_fresh) {
    for (int i = 0; i < kAngles; ++i) {
      out(i * 3 + 0) = state.gcf_scalar;
    }
  }
}

}  // namespace vf_robot_controller::perception
