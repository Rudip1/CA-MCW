// ClutterDetector — Phase 4.
//
// Local point density sensor: counts cached cloud points within
// `clutter_radius_` regardless of height. Distinguished from Gcf3D by
// scope — ClutterDetector reports "is the immediate vicinity messy?"
// while Gcf3D reports "is the body-level cylinder occupied?".
//
// Implements IGcf::query so it slots into GcfComposite alongside the
// other components.

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__GCF__CLUTTER_DETECTOR_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__GCF__CLUTTER_DETECTOR_HPP_

#include <array>
#include <memory>
#include <mutex>
#include <vector>

#include "vf_robot_controller/perception/gcf/i_gcf.hpp"

namespace vf_robot_controller::perception::gcf {

class ClutterDetector : public IGcf {
public:
  ClutterDetector() = default;
  explicit ClutterDetector(double clutter_radius)
  : clutter_radius_(clutter_radius) {}

  void setClutterRadius(double r) { clutter_radius_ = r; }
  void setSaturationCount(int n) { saturation_count_ = std::max(1, n); }

  void setPoints(std::shared_ptr<const std::vector<std::array<float, 3>>> pts);

  GcfCell query(double wx, double wy) const override;

private:
  double clutter_radius_{0.8};
  int saturation_count_{30};

  mutable std::mutex mu_;
  std::shared_ptr<const std::vector<std::array<float, 3>>> points_;
};

}  // namespace vf_robot_controller::perception::gcf

#endif  // VF_ROBOT_CONTROLLER__PERCEPTION__GCF__CLUTTER_DETECTOR_HPP_
