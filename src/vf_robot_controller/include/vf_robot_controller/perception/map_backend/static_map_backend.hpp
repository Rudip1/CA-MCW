// StaticMapBackend — reads pgm + yaml at startup, builds 2D occupancy grid,
// answers persistent-obstacle queries via per-ray DDA marching.
// Capabilities: { persistent_2d: true, topology: false, structure_3d: false }
// Phase 6 implementation.

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__MAP_BACKEND__STATIC_MAP_BACKEND_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__MAP_BACKEND__STATIC_MAP_BACKEND_HPP_

#include <memory>
#include <string>
#include "vf_robot_controller/perception/map_backend/i_map_backend.hpp"

namespace vf_robot_controller::perception {

class StaticMapBackend : public IMapBackend {
public:
  explicit StaticMapBackend(const std::string & yaml_path);
  ~StaticMapBackend() override;

  bool isAvailable() const override;
  BackendCapabilities capabilities() const override;
  std::vector<float> queryPersistentObstacles(
    const Pose2D & robot_pose,
    const std::vector<float> & angles,
    const std::vector<float> & radii) const override;
  std::optional<TopologicalFeatures> queryTopology(
    const Pose2D & robot_pose) const override;
  std::optional<StructuralFeatures3D> query3DStructure(
    const Pose2D & robot_pose) const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  bool available_{false};
};

}  // namespace vf_robot_controller::perception

#endif  // VF_ROBOT_CONTROLLER__PERCEPTION__MAP_BACKEND__STATIC_MAP_BACKEND_HPP_
