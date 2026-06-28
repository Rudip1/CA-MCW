// Gcf2D — 2D component of the Geometric Complexity Field. Phase 4.
//
// Pure-costmap GCF: at a query (wx, wy), inspects costmap cells in a
// circular neighbourhood of `radius_` and returns a [0,1] complexity that
// rises with mean inflation cost (proxy for nearby obstacles in the
// horizontal plane).
//
// The costmap is set externally per-cycle by gcf_node — Gcf2D never owns
// or subscribes. Thread-safe via a mutex around the pointer swap; the
// query path takes a const-ref snapshot and reads cell costs without
// further locking (costmap_2d's getCost is thread-safe for readers).

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__GCF__GCF_2D_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__GCF__GCF_2D_HPP_

#include <memory>
#include <mutex>

#include <nav2_costmap_2d/costmap_2d.hpp>

#include "vf_robot_controller/perception/gcf/i_gcf.hpp"

namespace vf_robot_controller::perception::gcf {

class Gcf2D : public IGcf {
public:
  Gcf2D() = default;
  explicit Gcf2D(double radius) : radius_(radius) {}

  void setRadius(double r) { radius_ = r; }
  void setCostmap(std::shared_ptr<nav2_costmap_2d::Costmap2D> costmap);

  GcfCell query(double wx, double wy) const override;

private:
  double radius_{2.0};
  mutable std::mutex mu_;
  std::shared_ptr<nav2_costmap_2d::Costmap2D> costmap_;
};

}  // namespace vf_robot_controller::perception::gcf

#endif  // VF_ROBOT_CONTROLLER__PERCEPTION__GCF__GCF_2D_HPP_
