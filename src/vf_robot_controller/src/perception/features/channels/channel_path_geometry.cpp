// PathGeometryChannel — 14 dims, pure math on global plan + odom. Phase 5.
//   [0]    distance_to_goal (m)
//   [1]    cumulative path length remaining (m)
//   [2]    nearest-path-segment cross-track error (m, signed)
//   [3..4] sin / cos of heading_error_to_lookahead
//   [5..7] curvature samples at 0.5, 1.0, 2.0 m lookahead
//   [8..10] heading samples at 0.5, 1.0, 2.0 m lookahead (wrapped to [-pi,pi])
//   [11]   path density (points / metre)
//   [12]   monotonic-progress flag (0/1) — has the closest point increased?
//   [13]   1 if path is empty / stale, 0 otherwise

#include "vf_robot_controller/perception/features/channels/channel_path_geometry.hpp"

#include <algorithm>
#include <cmath>

namespace vf_robot_controller::perception {

PathGeometryChannel::PathGeometryChannel() = default;

namespace {

// Squared distance between (rx, ry) and a path point.
inline float sqDist(float rx, float ry, const PathPoint & p)
{
  const float dx = rx - p.x;
  const float dy = ry - p.y;
  return dx * dx + dy * dy;
}

// Find index of the path point closest to (rx, ry).
size_t closestIndex(const std::vector<PathPoint> & path, float rx, float ry)
{
  size_t best = 0;
  float best_d2 = sqDist(rx, ry, path[0]);
  for (size_t i = 1; i < path.size(); ++i) {
    const float d2 = sqDist(rx, ry, path[i]);
    if (d2 < best_d2) { best_d2 = d2; best = i; }
  }
  return best;
}

// Find the path index whose cumulative arc-length from `from_idx` first
// reaches `target_arc`. Returns path.size()-1 if path ends first.
size_t lookaheadIndex(const std::vector<PathPoint> & path,
                      size_t from_idx, float target_arc)
{
  float arc = 0.0f;
  for (size_t i = from_idx + 1; i < path.size(); ++i) {
    const float dx = path[i].x - path[i - 1].x;
    const float dy = path[i].y - path[i - 1].y;
    arc += std::sqrt(dx * dx + dy * dy);
    if (arc >= target_arc) return i;
  }
  return path.size() - 1;
}

}  // namespace

void PathGeometryChannel::compute(
  const PerceptionState & state, Eigen::Ref<Eigen::VectorXf> out) const
{
  out.setZero();
  if (state.path.empty()) {
    out(13) = 1.0f;
    return;
  }

  const float rx = static_cast<float>(state.robot_pose.x);
  const float ry = static_cast<float>(state.robot_pose.y);
  const float rt = static_cast<float>(state.robot_pose.theta);

  const auto & path = state.path;
  const size_t closest = closestIndex(path, rx, ry);

  // [0] distance to goal: prefer the field if set, else compute from last pt.
  if (state.distance_to_goal > 0.0f) {
    out(0) = state.distance_to_goal;
  } else {
    const auto & end = path.back();
    out(0) = std::sqrt((end.x - rx) * (end.x - rx) + (end.y - ry) * (end.y - ry));
  }

  // [1] remaining cumulative arc length from closest point.
  float arc = 0.0f;
  for (size_t i = closest + 1; i < path.size(); ++i) {
    const float dx = path[i].x - path[i - 1].x;
    const float dy = path[i].y - path[i - 1].y;
    arc += std::sqrt(dx * dx + dy * dy);
  }
  out(1) = arc;

  // [2] cross-track error: signed distance to closest segment.
  // Approximation: project onto the segment between closest and next.
  if (closest + 1 < path.size()) {
    const auto & a = path[closest];
    const auto & b = path[closest + 1];
    const float dx = b.x - a.x;
    const float dy = b.y - a.y;
    const float len2 = dx * dx + dy * dy;
    if (len2 > 1e-9f) {
      // signed perpendicular distance using 2D cross product
      out(2) = ((rx - a.x) * dy - (ry - a.y) * dx) / std::sqrt(len2);
    }
  }

  // Lookahead samples at 0.5, 1.0, 2.0 m.
  const float lookaheads[3] = {0.5f, 1.0f, 2.0f};
  for (int k = 0; k < 3; ++k) {
    const size_t la_idx = lookaheadIndex(path, closest, lookaheads[k]);
    const float dx = path[la_idx].x - rx;
    const float dy = path[la_idx].y - ry;
    const float heading = std::atan2(dy, dx) - rt;
    // wrap to [-pi, pi]
    float wrapped = heading;
    while (wrapped >  3.14159265f) wrapped -= 6.28318531f;
    while (wrapped < -3.14159265f) wrapped += 6.28318531f;
    out(8 + k) = wrapped;

    // Curvature: discrete second-difference at the lookahead point.
    if (la_idx + 1 < path.size() && la_idx > 0) {
      const auto & p0 = path[la_idx - 1];
      const auto & p1 = path[la_idx];
      const auto & p2 = path[la_idx + 1];
      const float ax = p1.x - p0.x, ay = p1.y - p0.y;
      const float bx = p2.x - p1.x, by = p2.y - p1.y;
      const float cross = ax * by - ay * bx;
      const float la_seg = std::sqrt(ax * ax + ay * ay);
      out(5 + k) = (la_seg > 1e-6f) ? cross / (la_seg * la_seg * la_seg + 1e-6f) : 0.0f;
    }
  }

  // [3..4] sin/cos of immediate (0.5 m) heading error. Reuse computed [8].
  out(3) = std::sin(out(8));
  out(4) = std::cos(out(8));

  // [11] path density (points / metre over the remaining arc).
  out(11) = (arc > 1e-3f) ? static_cast<float>(path.size() - closest) / arc : 0.0f;

  // [12] monotonic-progress flag — bookkeeping handled by node, here we
  // can't see across cycles; conservatively report 1 when closest > 0.
  out(12) = (closest > 0) ? 1.0f : 0.0f;

  // [13] 0 (path is present).
  out(13) = 0.0f;
}

}  // namespace vf_robot_controller::perception
