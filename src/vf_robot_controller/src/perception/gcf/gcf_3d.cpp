#include "vf_robot_controller/perception/gcf/gcf_3d.hpp"

#include <algorithm>
#include <cmath>

namespace vf_robot_controller::perception::gcf {

Gcf3D::Gcf3D(double radius, double height_min, double height_max)
: radius_(radius), height_min_(height_min), height_max_(height_max) {}

void Gcf3D::setPoints(std::shared_ptr<const std::vector<std::array<float, 3>>> pts)
{
  std::lock_guard<std::mutex> lock(mu_);
  points_ = std::move(pts);
}

GcfCell Gcf3D::query(double wx, double wy) const
{
  GcfCell cell;
  std::shared_ptr<const std::vector<std::array<float, 3>>> snap;
  {
    std::lock_guard<std::mutex> lock(mu_);
    snap = points_;
  }
  if (!snap || snap->empty() || radius_ <= 0.0) return cell;

  const float r2 = static_cast<float>(radius_ * radius_);
  const float hmin = static_cast<float>(height_min_);
  const float hmax = static_cast<float>(height_max_);
  const float qx = static_cast<float>(wx);
  const float qy = static_cast<float>(wy);

  uint32_t hits = 0;
  for (const auto & p : *snap) {
    if (p[2] < hmin || p[2] > hmax) continue;
    const float dx = p[0] - qx;
    const float dy = p[1] - qy;
    if (dx * dx + dy * dy > r2) continue;
    ++hits;
  }

  const double sat = static_cast<double>(std::max(1, saturation_count_));
  cell.complexity = std::clamp(static_cast<double>(hits) / sat, 0.0, 1.0);
  cell.clearance_3d = 1.0 - cell.complexity;
  return cell;
}

}  // namespace vf_robot_controller::perception::gcf
