// Phase 0 stub for geometry utils. Phase 5 fills in.
#include "vf_robot_controller/perception/common/geometry_utils.hpp"
#include <cmath>

namespace vf_robot_controller::perception::geometry {

double wrapAngle(double a) {
  // TODO Phase 5: replace with proper implementation if needed
  while (a >  M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

}  // namespace vf_robot_controller::perception::geometry
