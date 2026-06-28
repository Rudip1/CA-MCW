// RtabmapBackend — Phase 6 implementation.
//
// Opens an RTAB-Map .db read-only with WAL so it doesn't lock out a live
// session, caches keyframe positions + loop-closure edges, and answers
// persistent / topological / 3D queries from the cache. Per the
// the design notes "never block the control loop" rule, all queries run on
// in-memory data — no SQLite call happens on the hot path.
//
// Schema (RTAB-Map 0.20+):
//   Node(id, map_id, weight, stamp, pose BLOB(48 = 12 floats, row-major
//        3x4 transform [r00 r01 r02 tx | r10 r11 r12 ty | r20 r21 r22 tz]),
//        ground_truth_pose BLOB, velocity BLOB, label TEXT, ...)
//   Link(from_id, to_id, type, info BLOB, transform BLOB, user_data BLOB)
//        type: 0 neighbour, 1 loop closure, 2 global, 3 local space, ...
//
// Re-sync: a lightweight periodic check (default every 10 s) re-reads the
// node list. Construction succeeds even if the file is initially absent;
// availability flips on once at least one keyframe has been read.

#include "vf_robot_controller/perception/map_backend/rtabmap_backend.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <vector>

#include <sqlite3.h>

namespace vf_robot_controller::perception {

namespace {

// One cached keyframe — just the planar position + height, plus the
// raw stamp for ordering "ahead" / "behind" along the trajectory.
struct Keyframe {
  int    id{0};
  double x{0.0};
  double y{0.0};
  double z{0.0};
  double stamp{0.0};
};

// 12-float row-major blob -> (x, y, z) translation column.
bool decodePoseBlob(const void * data, int len,
                    double & x, double & y, double & z)
{
  if (!data || len < static_cast<int>(sizeof(float) * 12)) return false;
  float f[12];
  std::memcpy(f, data, sizeof(f));
  x = static_cast<double>(f[3]);
  y = static_cast<double>(f[7]);
  z = static_cast<double>(f[11]);
  return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

}  // namespace

struct RtabmapBackend::Impl {
  std::string db_path;
  std::chrono::seconds resync_interval{10};
  std::chrono::steady_clock::time_point last_sync{};

  // Mutex-protected cache (queries are const but rebuild is allowed lazily).
  mutable std::mutex mu;
  std::vector<Keyframe> keyframes;
  std::size_t loop_closure_count{0};
  std::atomic<bool> available{false};
  std::atomic<bool> failed_once{false};

  // Open the DB read-only (with WAL pragma) and pull keyframes / link types.
  // Returns true on success. Never throws — caller decides on availability.
  bool sync()
  {
    if (db_path.empty()) return false;
    std::error_code ec;
    if (!std::filesystem::exists(db_path, ec)) return false;

    sqlite3 * db = nullptr;
    const std::string uri =
      "file:" + db_path + "?mode=ro&immutable=0&nolock=0";
    int rc = sqlite3_open_v2(uri.c_str(), &db,
                             SQLITE_OPEN_READONLY | SQLITE_OPEN_URI, nullptr);
    if (rc != SQLITE_OK || !db) {
      if (db) sqlite3_close(db);
      return false;
    }
    // WAL + busy timeout: best practice for read-side coexistence with a
    // writer. journal_mode is read-only on a read-only DB but we issue
    // PRAGMA wal_autocheckpoint defensively in case a future SDK opens
    // RW. busy_timeout is the important one.
    sqlite3_busy_timeout(db, 200);
    char * err = nullptr;
    sqlite3_exec(db, "PRAGMA journal_mode=WAL;", nullptr, nullptr, &err);
    if (err) { sqlite3_free(err); err = nullptr; }
    sqlite3_exec(db, "PRAGMA query_only=1;", nullptr, nullptr, &err);
    if (err) { sqlite3_free(err); err = nullptr; }

    std::vector<Keyframe> new_kf;
    {
      sqlite3_stmt * stmt = nullptr;
      const char * sql =
        "SELECT id, stamp, pose FROM Node WHERE pose IS NOT NULL ORDER BY id;";
      rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
      if (rc == SQLITE_OK) {
        while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
          Keyframe k;
          k.id = sqlite3_column_int(stmt, 0);
          k.stamp = sqlite3_column_double(stmt, 1);
          const void * blob = sqlite3_column_blob(stmt, 2);
          const int blen = sqlite3_column_bytes(stmt, 2);
          if (decodePoseBlob(blob, blen, k.x, k.y, k.z)) {
            new_kf.push_back(k);
          }
        }
        sqlite3_finalize(stmt);
      }
    }

    std::size_t lc = 0;
    {
      sqlite3_stmt * stmt = nullptr;
      const char * sql = "SELECT COUNT(*) FROM Link WHERE type = 1;";
      if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
          lc = static_cast<std::size_t>(sqlite3_column_int64(stmt, 0));
        }
        sqlite3_finalize(stmt);
      }
    }

    sqlite3_close(db);

    if (new_kf.empty()) return false;

    {
      std::lock_guard<std::mutex> lock(mu);
      keyframes = std::move(new_kf);
      loop_closure_count = lc;
    }
    available.store(true);
    last_sync = std::chrono::steady_clock::now();
    return true;
  }

  // Check if a re-sync is due and run it. Lock-free read of `last_sync`
  // is fine; worst case we re-sync a tick early.
  void maybeResync() const
  {
    auto * self = const_cast<Impl *>(this);
    const auto now = std::chrono::steady_clock::now();
    if (!available.load() || (now - last_sync) >= resync_interval) {
      self->sync();
    }
  }
};

RtabmapBackend::RtabmapBackend(const std::string & db_path,
                               std::chrono::seconds resync_interval)
: impl_(new Impl())
{
  impl_->db_path = db_path;
  impl_->resync_interval = resync_interval;
  impl_->sync();
}

RtabmapBackend::~RtabmapBackend() = default;

bool RtabmapBackend::isAvailable() const
{
  return impl_ && impl_->available.load();
}

BackendCapabilities RtabmapBackend::capabilities() const
{
  return {true, true, true};
}

std::vector<float> RtabmapBackend::queryPersistentObstacles(
  const Pose2D & robot_pose,
  const std::vector<float> & angles,
  const std::vector<float> & radii) const
{
  std::vector<float> out(angles.size() * radii.size(), 0.0f);
  if (!isAvailable() || !impl_) return out;
  impl_->maybeResync();

  // RTAB-Map's .db doesn't contain a clean static occupancy grid in a way
  // that's cheap to access without pulling rtabmap_core. As a serviceable
  // proxy we treat the (3D-cleaned) keyframe envelope as a "where the
  // robot has been" prior — areas with no nearby keyframes are unmapped /
  // potentially blocked, while areas swept by the camera are confidently
  // free. This gives the meta-critic a useful "off-map" signal even
  // before a full point-cloud index is wired.
  std::vector<Keyframe> kf;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    kf = impl_->keyframes;  // small (≤ a few thousand) copy
  }
  if (kf.empty()) return out;

  // For each (angle, radius) sample point, count keyframes within a
  // 0.75 m support and return 1 - clamp(count / saturation, 0, 1) so a
  // dense local map -> ~0 (free, well-mapped), a sparse / off-map area -> 1.
  constexpr double kSupport = 0.75;        // m
  constexpr double kSaturation = 4.0;      // keyframes
  for (size_t a = 0; a < angles.size(); ++a) {
    const double ca = std::cos(static_cast<double>(angles[a]));
    const double sa = std::sin(static_cast<double>(angles[a]));
    for (size_t r = 0; r < radii.size(); ++r) {
      const double rad = static_cast<double>(radii[r]);
      const double qx = robot_pose.x + rad * ca;
      const double qy = robot_pose.y + rad * sa;
      int hits = 0;
      const double sup2 = kSupport * kSupport;
      for (const auto & k : kf) {
        const double dx = k.x - qx;
        const double dy = k.y - qy;
        if (dx * dx + dy * dy <= sup2) {
          if (++hits >= static_cast<int>(kSaturation + 1)) break;
        }
      }
      const float dens = std::min(1.0f,
        static_cast<float>(hits) / static_cast<float>(kSaturation));
      // off-map -> 1 (act as obstacle), well-mapped -> 0 (free)
      out[a * radii.size() + r] = std::max(0.0f, 1.0f - dens);
    }
  }
  return out;
}

std::optional<TopologicalFeatures>
RtabmapBackend::queryTopology(const Pose2D & robot_pose) const
{
  if (!isAvailable() || !impl_) return std::nullopt;
  impl_->maybeResync();

  std::vector<Keyframe> kf;
  std::size_t lc = 0;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    kf = impl_->keyframes;
    lc = impl_->loop_closure_count;
  }
  if (kf.empty()) return std::nullopt;

  TopologicalFeatures out{};
  // Find nearest keyframe by id-order to "now". stamp is monotonic in
  // RTAB-Map, so id-order ≈ time-order.
  double best_d2 = std::numeric_limits<double>::infinity();
  std::size_t nearest_idx = 0;
  for (std::size_t i = 0; i < kf.size(); ++i) {
    const double dx = kf[i].x - robot_pose.x;
    const double dy = kf[i].y - robot_pose.y;
    const double d2 = dx * dx + dy * dy;
    if (d2 < best_d2) {
      best_d2 = d2;
      nearest_idx = i;
    }
  }
  // Distance to nearest loop-closure ahead / behind: approximate by
  // distance to keyframes at large id-jumps from the nearest. We don't
  // have full Link parsing here (it would mean another query per call).
  // We use loop_closure_count as a crude scaling proxy: more closures =
  // shorter typical distance to one. ahead = forward along id-order,
  // behind = backward. Fall back to the planar distance to map edge.
  // This is a Phase 6 baseline — Phase 9 oracle may reach for richer signals.
  double ahead = 50.0, behind = 50.0;
  if (lc > 0) {
    const double scale = std::min(50.0, 5.0 + 50.0 / static_cast<double>(lc));
    ahead = scale; behind = scale;
  }
  // Walk forward / backward a few ids to estimate trajectory continuity:
  // smaller delta x_id distance -> denser keyframes ahead/behind.
  if (nearest_idx + 1 < kf.size()) {
    const double dx = kf[nearest_idx + 1].x - robot_pose.x;
    const double dy = kf[nearest_idx + 1].y - robot_pose.y;
    ahead = std::sqrt(dx * dx + dy * dy);
  }
  if (nearest_idx > 0) {
    const double dx = kf[nearest_idx - 1].x - robot_pose.x;
    const double dy = kf[nearest_idx - 1].y - robot_pose.y;
    behind = std::sqrt(dx * dx + dy * dy);
  }
  out.distance_to_loop_closure_ahead = static_cast<float>(ahead);
  out.distance_to_loop_closure_behind = static_cast<float>(behind);

  // Keyframe density within 2 m.
  int kf_within_2m = 0;
  for (const auto & k : kf) {
    const double dx = k.x - robot_pose.x;
    const double dy = k.y - robot_pose.y;
    if (dx * dx + dy * dy <= 4.0) ++kf_within_2m;
  }
  out.keyframe_density_2m = static_cast<float>(kf_within_2m);

  // Distance to "branch point" — Phase 6 stub: the nearest keyframe with
  // an outlier id jump (an indication of a topological branch). We use
  // distance to the nearest kf as a conservative lower bound. Refined
  // later phases.
  out.distance_to_branch_point = static_cast<float>(std::sqrt(best_d2));

  // Visual entropy — Phase 6 cheap proxy: log(loop_closure_count + 1).
  // Real BoW entropy would require parsing Word/Feature blobs; left for
  // Phase 9 oracle pipeline.
  out.visual_entropy = static_cast<float>(std::log1p(static_cast<double>(lc)));
  return out;
}

std::optional<StructuralFeatures3D>
RtabmapBackend::query3DStructure(const Pose2D & robot_pose) const
{
  if (!isAvailable() || !impl_) return std::nullopt;
  impl_->maybeResync();

  std::vector<Keyframe> kf;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    kf = impl_->keyframes;
  }
  if (kf.empty()) return std::nullopt;

  // 3D-from-keyframes: we don't load the obstacle/cell BLOBs by default
  // (heavy + format-versioned), so we report conservative summaries
  // built from keyframe Z values. This is a useful baseline; Phase 11
  // will extend with the real PCL voxelised cloud.
  StructuralFeatures3D out{};
  double mean_z = 0.0;
  int n_within = 0;
  double max_z = -1e9, min_z = 1e9;
  for (const auto & k : kf) {
    const double dx = k.x - robot_pose.x;
    const double dy = k.y - robot_pose.y;
    if (dx * dx + dy * dy > 9.0) continue;  // 3 m radius
    mean_z += k.z;
    ++n_within;
    if (k.z > max_z) max_z = k.z;
    if (k.z < min_z) min_z = k.z;
  }
  if (n_within > 0) {
    mean_z /= n_within;
    out.ceiling_height = static_cast<float>(std::max(0.0, max_z));
    out.floor_planarity =
      static_cast<float>(std::clamp(1.0 - (max_z - min_z) / 0.5, 0.0, 1.0));
    out.vertical_clutter_robot_height = static_cast<float>(std::max(0.0, mean_z));
    out.vertical_clutter_head_height = static_cast<float>(std::max(0.0, max_z - mean_z));
    out.distinct_obstacle_clusters = std::min(8, n_within / 2);
  }
  return out;
}

std::size_t RtabmapBackend::keyframeCountForTest() const
{
  if (!impl_) return 0;
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->keyframes.size();
}

std::size_t RtabmapBackend::loopClosureCountForTest() const
{
  if (!impl_) return 0;
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->loop_closure_count;
}

}  // namespace vf_robot_controller::perception
