// IMapBackend — interface for swappable map sources.
//
// Implementations:
//   - StaticMapBackend  (pgm + yaml)
//   - RtabmapBackend    (SQLite .db, read-only WAL)
//   - CuvslamBackend    (stub for now)
//
// All implementations report capabilities so feature channels can degrade
// gracefully when an implementation does not support a query type.

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__MAP_BACKEND__I_MAP_BACKEND_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__MAP_BACKEND__I_MAP_BACKEND_HPP_

#include <optional>
#include <vector>
#include "vf_robot_controller/perception/common/types.hpp"

namespace vf_robot_controller::perception {

class IMapBackend {
public:
  virtual ~IMapBackend() = default;

  /// True if the backend has been initialized and can answer queries.
  virtual bool isAvailable() const = 0;

  /// Capabilities flags. FeatureExtractor reads this once at startup.
  virtual BackendCapabilities capabilities() const = 0;

  /// Query persistent (mapped) obstacles around the robot.
  /// Returns occupancy fraction per (angle, radius) sample.
  /// Length: angles.size() * radii.size().
  virtual std::vector<float> queryPersistentObstacles(
    const Pose2D & robot_pose,
    const std::vector<float> & angles,
    const std::vector<float> & radii) const = 0;

  /// Query topological features. nullopt if backend lacks topology support.
  virtual std::optional<TopologicalFeatures> queryTopology(
    const Pose2D & robot_pose) const = 0;

  /// Query 3D structural features. nullopt if backend lacks 3D support.
  virtual std::optional<StructuralFeatures3D> query3DStructure(
    const Pose2D & robot_pose) const = 0;
};

}  // namespace vf_robot_controller::perception

#endif  // VF_ROBOT_CONTROLLER__PERCEPTION__MAP_BACKEND__I_MAP_BACKEND_HPP_
