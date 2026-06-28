#include "vf_robot_controller/perception/gcf/clutter_detector.hpp"

#include <algorithm>

namespace vf_robot_controller::perception::gcf {

void ClutterDetector::setPoints(
  std::shared_ptr<const std::vector<std::array<float, 3>>> pts)
{
  std::lock_guard<std::mutex> lock(mu_);
  points_ = std::move(pts);
}

GcfCell ClutterDetector::query(double wx, double wy) const
{
  GcfCell cell;
  std::shared_ptr<const std::vector<std::array<float, 3>>> snap;
  {
    std::lock_guard<std::mutex> lock(mu_);
    snap = points_;
  }
  if (!snap || snap->empty() || clutter_radius_ <= 0.0) return cell;

  const float r2 = static_cast<float>(clutter_radius_ * clutter_radius_);
  const float qx = static_cast<float>(wx);
  const float qy = static_cast<float>(wy);

  uint32_t hits = 0;
  for (const auto & p : *snap) {
    const float dx = p[0] - qx;
    const float dy = p[1] - qy;
    if (dx * dx + dy * dy > r2) continue;
    ++hits;
  }

  const double sat = static_cast<double>(std::max(1, saturation_count_));
  cell.complexity = std::clamp(static_cast<double>(hits) / sat, 0.0, 1.0);
  cell.clutter_density = cell.complexity;
  return cell;
}

}  // namespace vf_robot_controller::perception::gcf
