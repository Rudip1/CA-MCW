// CorridorCritic — Phase 3.
// See header for the design rationale.

#include "vf_robot_controller/critics/corridor_critic.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Core>

#include "vf_robot_controller/controller/weight_cache.hpp"

namespace mppi::critics {

void CorridorCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(power_, "cost_power", 2);
  getParam(weight_, "cost_weight", 30.0f);
  getParam(max_lateral_dev_, "max_lateral_dev", 1.0f);
  getParam(threshold_to_consider_, "threshold_to_consider", 0.5f);
  getParam(trajectory_point_step_, "trajectory_point_step", 2);
  getParam(gcf_scale_min_, "gcf_scale_min", 0.5f);
  getParam(gcf_scale_max_, "gcf_scale_max", 2.0f);
  getParam(gcf_stale_seconds_, "gcf_stale_seconds", 1.0);

  yaml_weight_ = weight_;

  // Subscribe to /vf/gcf_state on the parent's executor. QoS is best-effort
  // depth 1: we want the latest value, no replay, no backlog.
  if (auto node = parent_.lock()) {
    clock_ = node->get_clock();
    last_gcf_stamp_ = clock_->now() - rclcpp::Duration::from_seconds(1e6);

    auto qos = rclcpp::QoS(1).best_effort();
    gcf_sub_ = node->create_subscription<std_msgs::msg::Float32>(
      "/vf/gcf_state", qos,
      [this](std_msgs::msg::Float32::ConstSharedPtr msg) {
        last_gcf_value_.store(msg->data);
        has_gcf_.store(true);
        std::lock_guard<std::mutex> lock(stamp_mu_);
        last_gcf_stamp_ = clock_->now();
      });
  }

  RCLCPP_INFO(
    logger_,
    "CorridorCritic initialised: power=%u weight=%.2f max_lateral_dev=%.2f "
    "gcf_scale=[%.2f,%.2f] stale_threshold=%.2fs",
    power_, weight_, max_lateral_dev_, gcf_scale_min_, gcf_scale_max_,
    gcf_stale_seconds_);
}

float CorridorCritic::currentGcfScale()
{
  if (!has_gcf_.load()) return 1.0f;

  rclcpp::Time stamp;
  {
    std::lock_guard<std::mutex> lock(stamp_mu_);
    stamp = last_gcf_stamp_;
  }

  if (clock_) {
    const auto age = (clock_->now() - stamp).seconds();
    if (age > gcf_stale_seconds_) {
      // Stale data — degrade to neutral. Phase 4 should restore freshness.
      return 1.0f;
    }
  }

  const float g = std::clamp(last_gcf_value_.load(), 0.0f, 1.0f);
  return gcf_scale_min_ + (gcf_scale_max_ - gcf_scale_min_) * g;
}

void CorridorCritic::score(CriticData & data)
{
  // Apply meta-critic weight injection. Mirrors WeightedCriticBase pattern
  // but inline because we don't inherit from an upstream critic.
  auto & cache = ::vf_robot_controller::WeightCache::instance();
  if (cache.isActive()) {
    auto m = cache.getMultiplier(getName());
    weight_ = yaml_weight_ * (m.has_value() ? *m : 1.0f);
  }

  const bool collect = cache.isCostCollectionActive();
  std::vector<float> before;
  if (collect) {
    before.assign(data.costs.cbegin(), data.costs.cend());
  }

  if (!enabled_ || data.path.x.size() < 2) {
    if (collect) {
      // Record an all-zero delta so downstream consumers see this critic.
      cache.recordDelta(getName(), std::vector<float>(data.costs.size(), 0.0f));
    }
    return;
  }

  // Bail out if the remaining path is too short to be meaningful.
  const size_t n_path = data.path.x.size();
  float path_length = 0.0f;
  for (size_t i = 1; i < n_path; ++i) {
    const float dx = data.path.x(i) - data.path.x(i - 1);
    const float dy = data.path.y(i) - data.path.y(i - 1);
    path_length += std::sqrt(dx * dx + dy * dy);
    if (path_length >= threshold_to_consider_) break;
  }
  if (path_length < threshold_to_consider_) {
    if (collect) {
      cache.recordDelta(getName(), std::vector<float>(data.costs.size(), 0.0f));
    }
    return;
  }

  const float gcf_scale = currentGcfScale();
  const float effective_weight = weight_ * gcf_scale;

  const size_t batch = data.trajectories.x.shape()[0];
  const size_t T = data.trajectories.x.shape()[1];
  const int step = std::max(1, trajectory_point_step_);

  Eigen::ArrayXf cost(batch);
  cost.setZero();

  for (size_t b = 0; b < batch; ++b) {
    float sum_norm_dev = 0.0f;
    unsigned int n = 0;
    for (size_t t = 0; t < T; t += static_cast<size_t>(step)) {
      const float tx = data.trajectories.x(b, t);
      const float ty = data.trajectories.y(b, t);
      // Brute-force nearest path point. n_path is typically <= 100, batch
      // is typically 4000, T strided ~28 → 11M comparisons / cycle worst
      // case, well under 5 ms on a modern CPU.
      float min_d2 = std::numeric_limits<float>::max();
      for (size_t p = 0; p < n_path; ++p) {
        const float dx = tx - data.path.x(p);
        const float dy = ty - data.path.y(p);
        const float d2 = dx * dx + dy * dy;
        if (d2 < min_d2) min_d2 = d2;
      }
      const float d = std::sqrt(min_d2);
      const float norm = std::min(d / max_lateral_dev_, 4.0f);
      sum_norm_dev += norm;
      ++n;
    }
    if (n > 0) {
      cost(b) = sum_norm_dev / static_cast<float>(n);
    }
  }

  Eigen::ArrayXf scaled = cost * effective_weight;
  Eigen::ArrayXf out(batch);
  if (power_ > 1u) {
    out = scaled.pow(static_cast<int>(power_));
  } else {
    out = scaled;
  }

  for (size_t b = 0; b < batch; ++b) {
    data.costs(b) += out(b);
  }

  if (collect) {
    std::vector<float> delta(data.costs.size());
    for (size_t i = 0; i < delta.size(); ++i) {
      delta[i] = data.costs(i) - before[i];
    }
    cache.recordDelta(getName(), std::move(delta));
  }
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(
  mppi::critics::CorridorCritic,
  mppi::critics::CriticFunction)
