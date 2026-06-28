// SlamPersistentChannel — Phase 6.
//
// Queries IMapBackend (provided via PerceptionState.map_backend) for
// topological / persistent-obstacle / 3D structural features. Cached
// with a TTL + pose-stability check so the 20 Hz feature loop never
// blocks waiting for a backend that takes several ms to answer (per
// the design notes "never block the control loop on perception" rule).
//
// Each block zero-fills if the backend doesn't support that capability.
// The two capability flags (slots 38, 39) tell the downstream MLP
// whether to trust the topology / 3D values.

#include "vf_robot_controller/perception/features/channels/channel_slam_persistent.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

#include "vf_robot_controller/perception/map_backend/i_map_backend.hpp"

namespace vf_robot_controller::perception {

namespace {

// Persistent-obstacle rosette layout: 8 evenly-spaced angles around the
// robot at two radii. Matches the CorridorCritic / GcfRosette geometry
// (16 polar samples), downsampled to 8 to leave room for the topology
// + 3D blocks within the 40-dim envelope.
constexpr int kAngles = 8;
constexpr int kRadii = 2;
constexpr float kRadii0 = 3.0f;  // metres — near
constexpr float kRadii1 = 5.0f;  // metres — far

constexpr float kTwoPi = 6.28318530718f;

}  // namespace

SlamPersistentChannel::SlamPersistentChannel()
: cache_(40)
{
  cache_.setZero();
}

void SlamPersistentChannel::invalidateCacheForTest()
{
  std::lock_guard<std::mutex> lock(mu_);
  cache_primed_ = false;
}

void SlamPersistentChannel::compute(
  const PerceptionState & state, Eigen::Ref<Eigen::VectorXf> out) const
{
  out.setZero();

  // No backend wired -> all zeros + capability flags = 0.
  if (!state.map_backend) return;

  const IMapBackend & backend = *state.map_backend;
  if (!backend.isAvailable()) {
    // Backend exists but isn't ready (e.g. RTAB-Map .db not yet found).
    // Leave capability flags 0 to signal "no real data here".
    return;
  }

  // Cache check: serve the previous result if the robot is essentially
  // stationary and the TTL hasn't expired. This is the staleness-tolerant
  // pattern used elsewhere in the perception pipeline.
  const auto now = std::chrono::steady_clock::now();
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (cache_primed_) {
      const double dx = state.robot_pose.x - last_x_;
      const double dy = state.robot_pose.y - last_y_;
      const double dtheta = std::abs(state.robot_pose.theta - last_theta_);
      const auto age = now - last_query_;
      if (age < cache_ttl_ &&
          std::sqrt(dx * dx + dy * dy) < cache_pose_eps_ &&
          dtheta < 0.10) {  // ~5 deg
        out = cache_;
        return;
      }
    }
  }

  const auto caps = backend.capabilities();

  // ── Block 1: topology (slots 0..11) ─────────────────────────────────
  if (caps.topology) {
    const auto topo_opt = backend.queryTopology(state.robot_pose);
    if (topo_opt) {
      const auto & t = *topo_opt;
      out(0) = t.distance_to_loop_closure_ahead;
      out(1) = t.distance_to_loop_closure_behind;
      out(2) = t.keyframe_density_2m;
      out(3) = t.distance_to_branch_point;
      out(4) = t.visual_entropy;
      // Normalised companions (so the MLP sees both raw + bounded forms).
      out(5) = std::tanh(t.distance_to_loop_closure_ahead / 10.0f);
      out(6) = std::tanh(t.distance_to_loop_closure_behind / 10.0f);
      out(7) = std::tanh(t.keyframe_density_2m / 8.0f);
      out(8) = std::tanh(t.distance_to_branch_point / 5.0f);
      out(9) = std::tanh(t.visual_entropy);
      out(10) = std::exp(-t.distance_to_branch_point / 2.0f);  // decays w/ dist
      out(11) = std::min(1.0f, t.keyframe_density_2m / 16.0f);  // saturating density
    }
  }

  // ── Block 2: persistent-obstacle rosette (slots 12..27) ─────────────
  if (caps.persistent_2d) {
    std::vector<float> angles(kAngles);
    for (int i = 0; i < kAngles; ++i) {
      angles[i] = static_cast<float>(state.robot_pose.theta) +
                  (kTwoPi * i) / kAngles;
    }
    const std::vector<float> radii = {kRadii0, kRadii1};
    const auto occ = backend.queryPersistentObstacles(
      state.robot_pose, angles, radii);
    // Layout in the channel: [angle][radius] flat -> out(12 + a*2 + r).
    for (int a = 0; a < kAngles; ++a) {
      for (int r = 0; r < kRadii; ++r) {
        const std::size_t idx = static_cast<std::size_t>(a) * kRadii +
                                static_cast<std::size_t>(r);
        if (idx < occ.size()) {
          out(12 + a * 2 + r) = std::clamp(occ[idx], 0.0f, 1.0f);
        }
      }
    }
  }

  // ── Block 3: 3D structure (slots 28..37) ────────────────────────────
  if (caps.structure_3d) {
    const auto s3_opt = backend.query3DStructure(state.robot_pose);
    if (s3_opt) {
      const auto & s = *s3_opt;
      out(28) = s.ceiling_height;
      out(29) = s.floor_planarity;
      out(30) = s.vertical_clutter_robot_height;
      out(31) = s.vertical_clutter_head_height;
      out(32) = static_cast<float>(s.distinct_obstacle_clusters);
      // Normalised companions
      out(33) = std::tanh(s.ceiling_height / 3.0f);
      out(34) = std::clamp(s.floor_planarity, 0.0f, 1.0f);
      out(35) = std::tanh(s.vertical_clutter_robot_height / 1.5f);
      out(36) = std::tanh(s.vertical_clutter_head_height / 1.5f);
      out(37) = std::min(1.0f,
                         static_cast<float>(s.distinct_obstacle_clusters) / 8.0f);
    }
  }

  // ── Block 4: capability flags (slots 38, 39) ────────────────────────
  out(38) = caps.topology ? 1.0f : 0.0f;
  out(39) = caps.structure_3d ? 1.0f : 0.0f;

  // Update cache.
  {
    std::lock_guard<std::mutex> lock(mu_);
    last_x_ = state.robot_pose.x;
    last_y_ = state.robot_pose.y;
    last_theta_ = state.robot_pose.theta;
    last_query_ = now;
    cache_ = out;
    cache_primed_ = true;
  }
}

}  // namespace vf_robot_controller::perception
