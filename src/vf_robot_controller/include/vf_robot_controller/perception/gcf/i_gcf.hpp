// IGcf — interface for the Geometric Complexity Field.
// Implementations: Gcf2D, Gcf3D, GcfComposite.
// Phase 4 implementation; interface locked in Phase 0.

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__GCF__I_GCF_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__GCF__I_GCF_HPP_

#include "vf_robot_controller/perception/common/types.hpp"

namespace vf_robot_controller::perception {

struct GcfCell {
  double complexity{0.0};       // [0,1] composite
  double clearance_2d{0.0};
  double clearance_3d{0.0};
  double clutter_density{0.0};
  bool   traversable{true};
};

class IGcf {
public:
  virtual ~IGcf() = default;
  // TODO Phase 4: full update/query API
  virtual GcfCell query(double wx, double wy) const = 0;
};

}  // namespace vf_robot_controller::perception

#endif  // VF_ROBOT_CONTROLLER__PERCEPTION__GCF__I_GCF_HPP_
