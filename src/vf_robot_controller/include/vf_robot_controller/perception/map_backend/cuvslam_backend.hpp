// CuvslamBackend — STUB. Reserved for Isaac cuVSLAM integration on Jetson Orin.
// Throws std::runtime_error on every method until Phase 11 (real-robot port).

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__MAP_BACKEND__CUVSLAM_BACKEND_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__MAP_BACKEND__CUVSLAM_BACKEND_HPP_

#include "vf_robot_controller/perception/map_backend/i_map_backend.hpp"

namespace vf_robot_controller::perception {

class CuvslamBackend : public IMapBackend {
public:
  CuvslamBackend();
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
};

}  // namespace vf_robot_controller::perception

#endif  // VF_ROBOT_CONTROLLER__PERCEPTION__MAP_BACKEND__CUVSLAM_BACKEND_HPP_
