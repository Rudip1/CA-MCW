// ReynoldsChannel — 4 dims, optional. Phase 5.
//
// Aggregates the four classical Reynolds primitives, each in [0,1]:
//   [0] separation     — proxied by gcf_scalar (high gcf == close obstacles)
//   [1] alignment      — heading-error magnitude vs. immediate path direction
//                        (sin(heading_err)^2 in case the path channel ran)
//   [2] cohesion       — clamp(1 - cross_track_err / 1.0m, 0, 1)
//   [3] goal_seeking   — clamp(1 - distance_to_goal / 5.0m, 0, 1)
//
// Reynolds is a derived/optional channel — it deliberately re-summarises
// signals that other channels expose at finer resolution, intended for
// ablation studies showing whether a 4-dim ethologically-motivated input
// adds anything over the raw channels. Behind a YAML flag.

#include "vf_robot_controller/perception/features/channels/channel_reynolds.hpp"

#include <algorithm>
#include <cmath>

namespace vf_robot_controller::perception {

ReynoldsChannel::ReynoldsChannel() = default;

void ReynoldsChannel::compute(
  const PerceptionState & state, Eigen::Ref<Eigen::VectorXf> out) const
{
  out.setZero();

  // Separation
  out(0) = std::clamp(state.gcf_scalar, 0.0f, 1.0f);

  // Alignment + cohesion — replay path math locally to avoid coupling to
  // the PathGeometryChannel output ordering.
  if (!state.path.empty()) {
    const float rx = static_cast<float>(state.robot_pose.x);
    const float ry = static_cast<float>(state.robot_pose.y);
    const float rt = static_cast<float>(state.robot_pose.theta);

    // Closest path point + cross-track error
    size_t closest = 0;
    float best_d2 = (rx - state.path[0].x) * (rx - state.path[0].x) +
                    (ry - state.path[0].y) * (ry - state.path[0].y);
    for (size_t i = 1; i < state.path.size(); ++i) {
      const float dx = rx - state.path[i].x;
      const float dy = ry - state.path[i].y;
      const float d2 = dx * dx + dy * dy;
      if (d2 < best_d2) { best_d2 = d2; closest = i; }
    }

    if (closest + 1 < state.path.size()) {
      const auto & a = state.path[closest];
      const auto & b = state.path[closest + 1];
      const float seg_dx = b.x - a.x;
      const float seg_dy = b.y - a.y;
      const float seg_heading = std::atan2(seg_dy, seg_dx);
      const float heading_err = seg_heading - rt;
      out(1) = std::clamp(std::cos(heading_err) * 0.5f + 0.5f, 0.0f, 1.0f);
    }

    out(2) = std::clamp(1.0f - std::sqrt(best_d2) / 1.0f, 0.0f, 1.0f);
  }

  // Goal-seeking
  out(3) = std::clamp(1.0f - state.distance_to_goal / 5.0f, 0.0f, 1.0f);
}

}  // namespace vf_robot_controller::perception
