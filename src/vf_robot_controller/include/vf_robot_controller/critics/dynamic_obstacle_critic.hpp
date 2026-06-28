// DynamicObstacleCritic — VF custom critic for nav2_mppi_controller. Phase 3.
//
// Penalises trajectories that pass through cells where the costmap has been
// rising — i.e. moving obstacles entering the robot's path. Maintains a
// short ring buffer of recent costmap snapshots (default 5) and at each
// candidate trajectory pose looks at the *delta* between the current
// costmap and the oldest snapshot. Positive delta == new obstacle here.
//
// Time-decay weighting: predicted poses earlier in the rollout horizon
// receive *higher* weight than later ones, on the principle that early
// avoidance is cheaper and far-future predictions are more uncertain.
// (The the design notes prompt described this as "early avoidance for far-future
// predictions" — implemented as exponential decay along the trajectory
// time axis with rate `time_decay_`.)
//
// **Source data.** Inherited `costmap_` member from CriticFunction.
// Sampled at the start of every score() call into a per-process ring buffer
// (singleton, since CriticFunction has no shared state across critic
// instances and we want a single history). Phase 4 may relocate this to a
// dedicated perception node, but for now the cost of grabbing 1024 cells
// (32 × 32 around the robot) per cycle is negligible.
//
// **Cost magnitude.** With raw costmap cells in [0, 254] (LETHAL_OBSTACLE
// is 254 in nav2), a single new obstacle in the trajectory path produces a
// per-pose delta around 100-250. Scaled by weight (default 0.4) and
// summed with time decay over ~28 sampled poses, per-trajectory cost lands
// in [0, ~1000] — well in the upstream band, between PathFollow and
// CostCritic.
//
// **Plugin namespace.** mppi::critics so upstream's CriticManager finds it.

#ifndef VF_ROBOT_CONTROLLER__CRITICS__DYNAMIC_OBSTACLE_CRITIC_HPP_
#define VF_ROBOT_CONTROLLER__CRITICS__DYNAMIC_OBSTACLE_CRITIC_HPP_

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "nav2_mppi_controller/critic_function.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"

namespace mppi::critics {

// One downsampled costmap window centred on the robot, retained over time.
struct CostmapSnapshot {
  // World-frame origin of the snapshot's cell (0,0).
  double origin_x{0.0};
  double origin_y{0.0};
  double resolution{0.05};
  unsigned int width{0};
  unsigned int height{0};
  std::vector<uint8_t> cells;  // size == width * height
};

class DynamicObstacleCritic : public CriticFunction {
public:
  DynamicObstacleCritic() = default;
  ~DynamicObstacleCritic() override = default;

  void initialize() override;
  void score(CriticData & data) override;

  // Test seam: inject a synthetic snapshot history.
  void seedHistoryForTest(std::deque<CostmapSnapshot> hist) {
    std::lock_guard<std::mutex> lock(history_mu_);
    history_ = std::move(hist);
  }

protected:
  // YAML-tunable parameters.
  unsigned int power_{1};
  float weight_{0.4f};
  float yaml_weight_{0.4f};
  size_t history_capacity_{5};
  int trajectory_point_step_{2};
  float time_decay_{1.5f};         // exp(-time_decay * t / horizon).
  uint8_t delta_threshold_{20};    // Costmap delta below this is ignored as noise.
  float window_radius_{2.5f};      // metres; half-side of the costmap window we keep.

  std::deque<CostmapSnapshot> history_;
  std::mutex history_mu_;

  std::shared_ptr<rclcpp::Clock> clock_;

  // Build a fresh snapshot of the area around (cx, cy) from the live
  // costmap. Returns false if the costmap is unavailable.
  bool sampleCostmap(double cx, double cy, CostmapSnapshot & out);

  // Look up a cell in the snapshot by world-frame xy. Returns 0 (free) when
  // the point is outside the snapshot's bounds.
  uint8_t lookup(const CostmapSnapshot & snap, float wx, float wy) const;
};

}  // namespace mppi::critics

#endif  // VF_ROBOT_CONTROLLER__CRITICS__DYNAMIC_OBSTACLE_CRITIC_HPP_
