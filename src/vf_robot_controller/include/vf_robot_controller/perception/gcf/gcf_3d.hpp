// Gcf3D — 3D component of the Geometric Complexity Field. Phase 4.
//
// Counts pointcloud hits inside a vertical cylinder around the query
// point — captures obstacles invisible to the 2D costmap (low overhangs,
// table edges, head-height clutter). Returns [0,1] density relative to
// `saturation_count_`.
//
// Points are set by gcf_node from the voxel-filtered cloud, in the same
// frame as the query coordinates (typically `odom`). Internal state is
// a thread-safe shared_ptr to a flat (x,y,z) vector.

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__GCF__GCF_3D_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__GCF__GCF_3D_HPP_

#include <array>
#include <memory>
#include <mutex>
#include <vector>

#include "vf_robot_controller/perception/gcf/i_gcf.hpp"

namespace vf_robot_controller::perception::gcf {

class Gcf3D : public IGcf {
public:
  Gcf3D() = default;
  Gcf3D(double radius, double height_min, double height_max);

  void setRadius(double r) { radius_ = r; }
  void setHeightBand(double hmin, double hmax) {
    height_min_ = hmin;
    height_max_ = hmax;
  }
  void setSaturationCount(int n) { saturation_count_ = std::max(1, n); }

  // Replace the cached cloud. Owned via shared_ptr so query() can read a
  // snapshot without locking the producer.
  void setPoints(std::shared_ptr<const std::vector<std::array<float, 3>>> pts);

  GcfCell query(double wx, double wy) const override;

private:
  double radius_{0.6};
  double height_min_{0.05};
  double height_max_{1.5};
  int saturation_count_{50};

  mutable std::mutex mu_;
  std::shared_ptr<const std::vector<std::array<float, 3>>> points_;
};

}  // namespace vf_robot_controller::perception::gcf

#endif  // VF_ROBOT_CONTROLLER__PERCEPTION__GCF__GCF_3D_HPP_
