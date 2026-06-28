// Common types shared across the perception pipeline.
// Phase 5: PerceptionState grew to hold all per-cycle inputs the feature
// channels read. The struct is built once per tick by feature_extractor_node
// from cached subscription data and passed into IFeatureChannel::compute().

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__COMMON__TYPES_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__COMMON__TYPES_HPP_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace vf_robot_controller::perception {

struct Pose2D {
  double x{0.0};
  double y{0.0};
  double theta{0.0};
};

struct BackendCapabilities {
  bool persistent_2d{false};
  bool topology{false};
  bool structure_3d{false};
};

struct TopologicalFeatures {
  float distance_to_loop_closure_ahead{0.0f};
  float distance_to_loop_closure_behind{0.0f};
  float keyframe_density_2m{0.0f};
  float distance_to_branch_point{0.0f};
  float visual_entropy{0.0f};
};

struct StructuralFeatures3D {
  float ceiling_height{0.0f};
  float floor_planarity{0.0f};
  float vertical_clutter_robot_height{0.0f};
  float vertical_clutter_head_height{0.0f};
  int   distinct_obstacle_clusters{0};
};

// Minimal forward declaration so we can hold a shared_ptr without pulling
// the full nav2_costmap_2d header into every channel (channels that don't
// need it shouldn't pay the include cost).
}  // namespace vf_robot_controller::perception
namespace nav2_costmap_2d { class Costmap2D; }
namespace vf_robot_controller::perception::gcf { class GcfComposite; }
namespace vf_robot_controller::perception { class IMapBackend; }

namespace vf_robot_controller::perception {

// Path point — minimal representation of the global plan. Channel code
// shouldn't need the full nav_msgs::Path; we project to 2D here.
struct PathPoint {
  float x{0.0f};
  float y{0.0f};
};

// One snapshot of per-critic costs from /vf/per_critic_costs.
struct CriticCostSample {
  // Indexed by critic order in the YAML (length up to 11). Values are the
  // per-critic delta summed across trajectories.
  std::vector<float> costs;
};

// Snapshot of perception state passed into feature channels.
// Built once per cycle by feature_extractor_node.
struct PerceptionState {
  // Robot state (always populated).
  Pose2D robot_pose;
  Eigen::Vector3f velocity{0.0f, 0.0f, 0.0f};   // vx, vy, wz
  Eigen::Vector3f acceleration{0.0f, 0.0f, 0.0f};  // ax, ay, alpha

  // Global plan (may be empty before first SetPlan).
  std::vector<PathPoint> path;
  float distance_to_goal{0.0f};

  // GCF cache. `gcf_scalar` is the at-pose value from /vf/gcf_state;
  // `gcf_composite` (optional) lets channels query at multiple angles
  // without duplicating subscribers.
  float gcf_scalar{0.0f};
  bool gcf_fresh{false};
  std::shared_ptr<gcf::GcfComposite> gcf_composite;

  // Costmap snapshots. now == this cycle, prev == one cycle ago (for
  // obstacle_dynamics deltas). nullptr when uninitialised.
  std::shared_ptr<nav2_costmap_2d::Costmap2D> costmap_now;
  std::shared_ptr<nav2_costmap_2d::Costmap2D> costmap_prev;

  // Context. Defaults to UNKNOWN (255).
  uint8_t context_id{255};

  // Critic-cost history: most recent first cycle has costs.size() >= 1.
  // Up to history_capacity samples; channel code zero-fills missing ones.
  std::vector<CriticCostSample> critic_history;

  // Phase 6: persistent-map backend (RTAB-Map / static / cuvslam). Set once
  // by feature_extractor_node at startup; channels read it during compute().
  // Null when no backend has been configured — channels zero-fill silently.
  std::shared_ptr<IMapBackend> map_backend;
};

}  // namespace vf_robot_controller::perception

#endif  // VF_ROBOT_CONTROLLER__PERCEPTION__COMMON__TYPES_HPP_
