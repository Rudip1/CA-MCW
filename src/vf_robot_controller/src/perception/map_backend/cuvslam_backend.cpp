// src/perception/map_backend/cuvslam_backend.cpp — Phase 0 stub.
// TODO Phase 6: implement.
#include "vf_robot_controller/perception/map_backend/cuvslam_backend.hpp"
#include <stdexcept>

namespace vf_robot_controller::perception {

CuvslamBackend::CuvslamBackend() = default;
bool CuvslamBackend::isAvailable() const { return false; }
BackendCapabilities CuvslamBackend::capabilities() const {
  return {false, false, false};
}
std::vector<float> CuvslamBackend::queryPersistentObstacles(
  const Pose2D &, const std::vector<float> &, const std::vector<float> &) const {
  throw std::runtime_error("CuvslamBackend not implemented (Phase 11)");
}
std::optional<TopologicalFeatures> CuvslamBackend::queryTopology(const Pose2D &) const {
  return std::nullopt;
}
std::optional<StructuralFeatures3D> CuvslamBackend::query3DStructure(const Pose2D &) const {
  return std::nullopt;
}

}  // namespace vf_robot_controller::perception
