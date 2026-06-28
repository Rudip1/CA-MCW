// ObstacleDynamicsChannel — 16 dims = 8 angular bins × 2 stats. Phase 5.
//
// Compares costmap_now vs costmap_prev. For each cell within `radius_`
// of the robot, classifies the bin index by angle-relative-to-robot-heading
// and accumulates:
//   - bin[i*2 + 0]: max positive delta (new obstacle appearing)
//   - bin[i*2 + 1]: max negative delta (obstacle leaving)
//
// If either snapshot is missing, output is zero. This is the live feed the
// DynamicObstacleCritic uses for its scoring; here we summarise it as a
// feature for the meta-critic.

#include "vf_robot_controller/perception/features/channels/channel_obstacle_dynamics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <nav2_costmap_2d/costmap_2d.hpp>

namespace vf_robot_controller::perception {

namespace {
constexpr int kBins = 8;
constexpr float kRadius = 2.0f;       // metres
constexpr float kTwoPi = 6.28318530718f;
}  // namespace

ObstacleDynamicsChannel::ObstacleDynamicsChannel() = default;

void ObstacleDynamicsChannel::compute(
  const PerceptionState & state, Eigen::Ref<Eigen::VectorXf> out) const
{
  out.setZero();
  if (!state.costmap_now || !state.costmap_prev) return;

  const auto & cm_now = *state.costmap_now;
  const auto & cm_prev = *state.costmap_prev;

  const double res_now = cm_now.getResolution();
  const double res_prev = cm_prev.getResolution();
  if (res_now <= 0.0 || res_prev <= 0.0) return;

  const double rx = state.robot_pose.x;
  const double ry = state.robot_pose.y;
  const float rt = static_cast<float>(state.robot_pose.theta);

  unsigned int mx_c = 0, my_c = 0;
  if (!cm_now.worldToMap(rx, ry, mx_c, my_c)) return;

  const int half = std::max(1, static_cast<int>(std::ceil(kRadius / res_now)));
  const int sx = static_cast<int>(cm_now.getSizeInCellsX());
  const int sy = static_cast<int>(cm_now.getSizeInCellsY());
  const int x0 = std::max(0, static_cast<int>(mx_c) - half);
  const int x1 = std::min(sx - 1, static_cast<int>(mx_c) + half);
  const int y0 = std::max(0, static_cast<int>(my_c) - half);
  const int y1 = std::min(sy - 1, static_cast<int>(my_c) + half);

  for (int j = y0; j <= y1; ++j) {
    for (int i = x0; i <= x1; ++i) {
      // World coords of this cell centre
      double wx, wy;
      cm_now.mapToWorld(static_cast<unsigned int>(i),
                        static_cast<unsigned int>(j), wx, wy);
      const double dx = wx - rx;
      const double dy = wy - ry;
      const double r2 = dx * dx + dy * dy;
      if (r2 > kRadius * kRadius || r2 < 1e-6) continue;

      // Costmap delta. Lookup in prev by world coords.
      const uint8_t cur = cm_now.getCost(static_cast<unsigned int>(i),
                                          static_cast<unsigned int>(j));
      unsigned int pi = 0, pj = 0;
      uint8_t prev = 0;
      if (cm_prev.worldToMap(wx, wy, pi, pj)) {
        prev = cm_prev.getCost(pi, pj);
      }
      const int delta = static_cast<int>(cur) - static_cast<int>(prev);
      if (delta == 0) continue;

      // Bin by angle relative to robot heading.
      float angle = std::atan2(static_cast<float>(dy), static_cast<float>(dx)) - rt;
      while (angle <  0.0f) angle += kTwoPi;
      while (angle >= kTwoPi) angle -= kTwoPi;
      const int bin = static_cast<int>(angle * kBins / kTwoPi) % kBins;

      const float fdelta = static_cast<float>(delta);
      if (fdelta > 0.0f) {
        out(bin * 2 + 0) = std::max(out(bin * 2 + 0), fdelta);
      } else {
        out(bin * 2 + 1) = std::min(out(bin * 2 + 1), fdelta);
      }
    }
  }
}

}  // namespace vf_robot_controller::perception
