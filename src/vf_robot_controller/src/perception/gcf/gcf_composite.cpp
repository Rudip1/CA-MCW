#include "vf_robot_controller/perception/gcf/gcf_composite.hpp"

#include <algorithm>

namespace vf_robot_controller::perception::gcf {

GcfComposite::GcfComposite(
  std::shared_ptr<Gcf2D> gcf_2d,
  std::shared_ptr<Gcf3D> gcf_3d,
  std::shared_ptr<ClutterDetector> clutter,
  GcfCompositeWeights weights)
: gcf_2d_(std::move(gcf_2d)),
  gcf_3d_(std::move(gcf_3d)),
  clutter_(std::move(clutter)),
  weights_(weights)
{
}

GcfCell GcfComposite::query(double wx, double wy) const
{
  GcfCell out;
  double total_w = 0.0, sum = 0.0;

  if (gcf_2d_ && weights_.w_2d > 0.0) {
    auto c = gcf_2d_->query(wx, wy);
    sum += weights_.w_2d * c.complexity;
    total_w += weights_.w_2d;
    out.clearance_2d = c.clearance_2d;
    out.traversable = out.traversable && c.traversable;
  }
  if (gcf_3d_ && weights_.w_volumetric > 0.0) {
    auto c = gcf_3d_->query(wx, wy);
    sum += weights_.w_volumetric * c.complexity;
    total_w += weights_.w_volumetric;
    out.clearance_3d = c.clearance_3d;
  }
  if (clutter_ && weights_.w_clutter > 0.0) {
    auto c = clutter_->query(wx, wy);
    sum += weights_.w_clutter * c.complexity;
    total_w += weights_.w_clutter;
    out.clutter_density = c.clutter_density;
  }

  out.complexity = total_w > 0.0 ? std::clamp(sum / total_w, 0.0, 1.0) : 0.0;
  return out;
}

}  // namespace vf_robot_controller::perception::gcf
