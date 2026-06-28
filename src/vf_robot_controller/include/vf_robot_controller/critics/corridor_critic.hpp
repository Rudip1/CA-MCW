// CorridorCritic — VF custom critic for nav2_mppi_controller. Phase 3.
//
// Penalises trajectories that drift laterally off the global plan, with the
// penalty *amplified* in tight corridors (high GCF density) and *softened*
// in open spaces. This is the "stay in the channel" signal that path
// alignment alone does not provide — PathAlignCritic measures error along
// the whole trajectory but uses a scalar weight; CorridorCritic injects a
// context multiplier that grows when the world geometrically constrains
// the robot.
//
// **GCF dependency.** Subscribes to /vf/gcf_state (std_msgs/Float32, scalar
// in [0, 1] interpreted as corridor-tightness / clutter density). Phase 4
// will land the publisher. Until then, the topic is silent and the critic
// uses gcf_scale == 1.0 — meaning it still produces a non-zero contribution
// based on cross-track error alone, just without context modulation.
//
// **Cost magnitude.** Per-trajectory cost is in [0, weight * gcf_max_scale],
// with weight default 30.0 and gcf_max_scale 2.0 → max ~60 before pow.
// With cost_power == 2 and typical lateral errors of 0.3-0.8 m relative to
// max_lateral_dev=1.0, mean cost ends up in [10, 60] for moderate-quality
// trajectories — same band as upstream PathAlignCritic.
//
// **Plugin namespace.** mppi::critics so upstream's CriticManager finds it.
// See weighted_critic_wrapper.hpp for the rationale.

#ifndef VF_ROBOT_CONTROLLER__CRITICS__CORRIDOR_CRITIC_HPP_
#define VF_ROBOT_CONTROLLER__CRITICS__CORRIDOR_CRITIC_HPP_

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <std_msgs/msg/float32.hpp>

#include "nav2_mppi_controller/critic_function.hpp"

namespace mppi::critics {

class CorridorCritic : public CriticFunction {
public:
  CorridorCritic() = default;
  ~CorridorCritic() override = default;

  void initialize() override;
  void score(CriticData & data) override;

  // Test seam: inject a GCF value bypassing the ROS subscription.
  void setGcfForTest(float v) {
    last_gcf_value_.store(v);
    has_gcf_.store(true);
    std::lock_guard<std::mutex> lock(stamp_mu_);
    last_gcf_stamp_ = clock_ ? clock_->now() : rclcpp::Time(0, 0, RCL_ROS_TIME);
  }

protected:
  // YAML-tunable parameters.
  unsigned int power_{2};
  float weight_{30.0f};
  float yaml_weight_{30.0f};        // Snapshot for meta-critic injection.
  float max_lateral_dev_{1.0f};      // metres; normalises cross-track error.
  float threshold_to_consider_{0.5f};  // metres of remaining path to bother scoring.
  int trajectory_point_step_{2};
  float gcf_scale_min_{0.5f};        // gcf == 0 → multiplier 0.5 (open).
  float gcf_scale_max_{2.0f};        // gcf == 1 → multiplier 2.0 (tight).
  double gcf_stale_seconds_{1.0};

  // Subscriber state — pure cache, never blocks the control loop.
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr gcf_sub_;
  std::atomic<float> last_gcf_value_{0.0f};
  std::atomic<bool> has_gcf_{false};
  rclcpp::Time last_gcf_stamp_;
  std::mutex stamp_mu_;

  std::shared_ptr<rclcpp::Clock> clock_;

  float currentGcfScale();
};

}  // namespace mppi::critics

#endif  // VF_ROBOT_CONTROLLER__CRITICS__CORRIDOR_CRITIC_HPP_
