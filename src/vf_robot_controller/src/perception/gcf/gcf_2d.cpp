#include "vf_robot_controller/perception/gcf/gcf_2d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <nav2_costmap_2d/cost_values.hpp>

namespace vf_robot_controller::perception::gcf {

void Gcf2D::setCostmap(std::shared_ptr<nav2_costmap_2d::Costmap2D> costmap)
{
  std::lock_guard<std::mutex> lock(mu_);
  costmap_ = std::move(costmap);
}

GcfCell Gcf2D::query(double wx, double wy) const
{
  GcfCell cell;
  std::shared_ptr<nav2_costmap_2d::Costmap2D> snap;
  {
    std::lock_guard<std::mutex> lock(mu_);
    snap = costmap_;
  }
  if (!snap || radius_ <= 0.0) return cell;

  const double res = snap->getResolution();
  if (res <= 0.0) return cell;

  unsigned int mx_c = 0, my_c = 0;
  if (!snap->worldToMap(wx, wy, mx_c, my_c)) {
    return cell;
  }

  const int half = std::max(1, static_cast<int>(std::ceil(radius_ / res)));
  const int sx = static_cast<int>(snap->getSizeInCellsX());
  const int sy = static_cast<int>(snap->getSizeInCellsY());
  const int x0 = std::max(0, static_cast<int>(mx_c) - half);
  const int x1 = std::min(sx - 1, static_cast<int>(mx_c) + half);
  const int y0 = std::max(0, static_cast<int>(my_c) - half);
  const int y1 = std::min(sy - 1, static_cast<int>(my_c) + half);

  const double r_cells_sq = static_cast<double>(half) * static_cast<double>(half);

  uint64_t sum = 0;
  uint32_t n = 0;
  uint32_t lethal = 0;
  for (int j = y0; j <= y1; ++j) {
    for (int i = x0; i <= x1; ++i) {
      const double dx = static_cast<double>(i) - static_cast<double>(mx_c);
      const double dy = static_cast<double>(j) - static_cast<double>(my_c);
      if (dx * dx + dy * dy > r_cells_sq) continue;
      const uint8_t c = snap->getCost(
        static_cast<unsigned int>(i), static_cast<unsigned int>(j));
      sum += c;
      ++n;
      if (c == nav2_costmap_2d::LETHAL_OBSTACLE) ++lethal;
    }
  }

  if (n == 0) return cell;

  const double mean_cost = static_cast<double>(sum) / static_cast<double>(n);
  // Saturation at 200/254 — well into inflation territory, below the
  // inscribed-obstacle plateau, so a wall cluster yields complexity ≈ 1.
  cell.complexity = std::clamp(mean_cost / 200.0, 0.0, 1.0);
  cell.clearance_2d = std::max(0.0, 1.0 - mean_cost / 254.0);
  cell.traversable = (lethal == 0);
  return cell;
}

}  // namespace vf_robot_controller::perception::gcf
