// RtabmapBackend — reads RTAB-Map .db (SQLite WAL, read-only).
// Capabilities: { persistent_2d, topology, structure_3d } — all supported.
//
// Phase 6 implementation. Opens the database with SQLITE_OPEN_READONLY and
// PRAGMA journal_mode=WAL so it can run alongside a live mapping session
// (rtabmap_loc / rtabmap holds an exclusive write transaction; we read).
//
// On open we cache:
//   - keyframe poses (Node.pose, 12-float 3x4 row-major blob)
//   - graph edges from Link (type 0 = neighbour, type 1 = loop closure)
//
// We periodically re-sync (default every 10 s) to pick up new keyframes
// added by RTAB. Re-sync is best-effort: if the DB is busy or missing we
// keep the previous cache and log once.
//
// Falling back: when the .db cannot be opened (missing file, locked, or
// schema mismatch), `isAvailable()` reports false and queries return
// zero-filled output. Higher-level code should chain a StaticMapBackend
// behind this one if it wants graceful 2D-only fallback (see
// `feature_extractor_node`).

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__MAP_BACKEND__RTABMAP_BACKEND_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__MAP_BACKEND__RTABMAP_BACKEND_HPP_

#include <chrono>
#include <memory>
#include <string>

#include "vf_robot_controller/perception/map_backend/i_map_backend.hpp"

namespace vf_robot_controller::perception {

class RtabmapBackend : public IMapBackend {
public:
  explicit RtabmapBackend(const std::string & db_path,
                          std::chrono::seconds resync_interval =
                            std::chrono::seconds(10));
  ~RtabmapBackend() override;

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

  // Test-only: keyframe count after the most recent (re)sync.
  std::size_t keyframeCountForTest() const;
  std::size_t loopClosureCountForTest() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vf_robot_controller::perception

#endif  // VF_ROBOT_CONTROLLER__PERCEPTION__MAP_BACKEND__RTABMAP_BACKEND_HPP_
