// GcfComposite — weighted blend of Gcf2D, Gcf3D, ClutterDetector. Phase 4.
//
// Holds shared_ptrs to the three concrete IGcf impls and combines their
// `complexity` outputs with weights from perception.yaml. Returns a
// composite GcfCell whose `complexity` field is the [0,1] scalar that
// gcf_node publishes on /vf/gcf_state.
//
// The component shared_ptrs are non-owning of the underlying state
// (costmap / pointcloud) — gcf_node sets that on each component before
// querying.

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__GCF__GCF_COMPOSITE_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__GCF__GCF_COMPOSITE_HPP_

#include <memory>

#include "vf_robot_controller/perception/gcf/clutter_detector.hpp"
#include "vf_robot_controller/perception/gcf/gcf_2d.hpp"
#include "vf_robot_controller/perception/gcf/gcf_3d.hpp"
#include "vf_robot_controller/perception/gcf/i_gcf.hpp"

namespace vf_robot_controller::perception::gcf {

struct GcfCompositeWeights {
  double w_2d{0.4};
  double w_clutter{0.3};
  double w_volumetric{0.3};
};

class GcfComposite : public IGcf {
public:
  GcfComposite(
    std::shared_ptr<Gcf2D> gcf_2d,
    std::shared_ptr<Gcf3D> gcf_3d,
    std::shared_ptr<ClutterDetector> clutter,
    GcfCompositeWeights weights);

  void setWeights(const GcfCompositeWeights & w) { weights_ = w; }
  GcfCell query(double wx, double wy) const override;

  std::shared_ptr<Gcf2D> gcf2d() const { return gcf_2d_; }
  std::shared_ptr<Gcf3D> gcf3d() const { return gcf_3d_; }
  std::shared_ptr<ClutterDetector> clutter() const { return clutter_; }

private:
  std::shared_ptr<Gcf2D> gcf_2d_;
  std::shared_ptr<Gcf3D> gcf_3d_;
  std::shared_ptr<ClutterDetector> clutter_;
  GcfCompositeWeights weights_;
};

}  // namespace vf_robot_controller::perception::gcf

#endif  // VF_ROBOT_CONTROLLER__PERCEPTION__GCF__GCF_COMPOSITE_HPP_
