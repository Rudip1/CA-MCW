// DynamicObstacleCritic — Phase 3.
// See header for the design rationale.

#include "vf_robot_controller/critics/dynamic_obstacle_critic.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "vf_robot_controller/controller/weight_cache.hpp"

namespace mppi::critics {

void DynamicObstacleCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(power_, "cost_power", 1);
  getParam(weight_, "cost_weight", 0.4f);
  int hist_cap_int = static_cast<int>(history_capacity_);
  getParam(hist_cap_int, "history_capacity", 5);
  history_capacity_ = static_cast<size_t>(std::max(1, hist_cap_int));
  getParam(trajectory_point_step_, "trajectory_point_step", 2);
  getParam(time_decay_, "time_decay", 1.5f);
  int dt = delta_threshold_;
  getParam(dt, "delta_threshold", static_cast<int>(delta_threshold_));
  delta_threshold_ = static_cast<uint8_t>(std::clamp(dt, 0, 254));
  getParam(window_radius_, "window_radius", 2.5f);

  yaml_weight_ = weight_;

  if (auto node = parent_.lock()) {
    clock_ = node->get_clock();
  }

  RCLCPP_INFO(
    logger_,
    "DynamicObstacleCritic initialised: weight=%.2f history=%zu decay=%.2f "
    "delta_thresh=%u radius=%.2fm",
    weight_, history_capacity_, time_decay_,
    static_cast<unsigned>(delta_threshold_), window_radius_);
}

bool DynamicObstacleCritic::sampleCostmap(
  double cx, double cy, CostmapSnapshot & out)
{
  if (!costmap_) return false;

  out.resolution = costmap_->getResolution();
  if (out.resolution <= 0.0) return false;

  // Build a window of side (2 * window_radius_) around (cx, cy), clipped
  // to costmap bounds.
  unsigned int mx_lo = 0, my_lo = 0, mx_hi = 0, my_hi = 0;
  costmap_->worldToMap(cx - window_radius_, cy - window_radius_, mx_lo, my_lo);
  if (!costmap_->worldToMap(cx + window_radius_, cy + window_radius_, mx_hi, my_hi)) {
    // upper bound clamped — get raw costmap dims
    mx_hi = costmap_->getSizeInCellsX();
    my_hi = costmap_->getSizeInCellsY();
  }

  if (mx_hi <= mx_lo || my_hi <= my_lo) return false;

  out.width = mx_hi - mx_lo;
  out.height = my_hi - my_lo;
  // Anchor the snapshot's origin at the world coords of cell (mx_lo, my_lo)
  // so lookup() can do straight cell math.
  costmap_->mapToWorld(mx_lo, my_lo, out.origin_x, out.origin_y);

  out.cells.assign(static_cast<size_t>(out.width) * out.height, 0);
  for (unsigned int j = 0; j < out.height; ++j) {
    for (unsigned int i = 0; i < out.width; ++i) {
      out.cells[j * out.width + i] = costmap_->getCost(mx_lo + i, my_lo + j);
    }
  }
  return true;
}

uint8_t DynamicObstacleCritic::lookup(
  const CostmapSnapshot & snap, float wx, float wy) const
{
  if (snap.cells.empty() || snap.resolution <= 0.0) return 0;
  const float dx = static_cast<float>(wx - snap.origin_x);
  const float dy = static_cast<float>(wy - snap.origin_y);
  if (dx < 0.0f || dy < 0.0f) return 0;
  const unsigned int i = static_cast<unsigned int>(dx / snap.resolution);
  const unsigned int j = static_cast<unsigned int>(dy / snap.resolution);
  if (i >= snap.width || j >= snap.height) return 0;
  return snap.cells[j * snap.width + i];
}

void DynamicObstacleCritic::score(CriticData & data)
{
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

  if (!enabled_) {
    if (collect) cache.recordDelta(getName(), std::vector<float>(data.costs.size(), 0.0f));
    return;
  }

  // Sample the live costmap into a snapshot centred on the first trajectory
  // pose (== current robot pose).
  const size_t batch = data.trajectories.x.shape()[0];
  const size_t T = data.trajectories.x.shape()[1];
  if (batch == 0 || T == 0) {
    if (collect) cache.recordDelta(getName(), std::vector<float>(data.costs.size(), 0.0f));
    return;
  }

  const double cx = static_cast<double>(data.trajectories.x(0, 0));
  const double cy = static_cast<double>(data.trajectories.y(0, 0));

  // Try to push a fresh snapshot from the live costmap. When the costmap
  // is absent (typical in unit tests where the test seeded history via
  // seedHistoryForTest), we fall through and use whatever the seam left.
  std::deque<CostmapSnapshot> history_copy;
  {
    std::lock_guard<std::mutex> lock(history_mu_);
    CostmapSnapshot fresh;
    if (sampleCostmap(cx, cy, fresh)) {
      history_.push_back(std::move(fresh));
      while (history_.size() > history_capacity_) history_.pop_front();
    }
    history_copy = history_;
  }

  if (history_copy.size() < 2) {
    if (collect) cache.recordDelta(getName(), std::vector<float>(data.costs.size(), 0.0f));
    return;
  }

  const CostmapSnapshot & current = history_copy.back();
  const CostmapSnapshot & oldest = history_copy.front();

  const int step = std::max(1, trajectory_point_step_);
  const float horizon = static_cast<float>(T - 1);
  std::vector<float> cost(batch, 0.0f);

  for (size_t b = 0; b < batch; ++b) {
    float traj_cost = 0.0f;
    for (size_t t = 0; t < T; t += static_cast<size_t>(step)) {
      const float tx = data.trajectories.x(b, t);
      const float ty = data.trajectories.y(b, t);
      const uint8_t cur = lookup(current, tx, ty);
      const uint8_t old = lookup(oldest, tx, ty);
      const int delta = static_cast<int>(cur) - static_cast<int>(old);
      if (delta < static_cast<int>(delta_threshold_)) continue;

      // Time decay: earlier trajectory points weighted more.
      const float frac = (horizon > 0.0f) ? static_cast<float>(t) / horizon : 0.0f;
      const float decay = std::exp(-time_decay_ * frac);
      traj_cost += static_cast<float>(delta) * decay;
    }
    cost[b] = traj_cost;
  }

  for (size_t b = 0; b < batch; ++b) {
    float c = cost[b] * weight_;
    if (power_ > 1u) c = std::pow(c, static_cast<int>(power_));
    data.costs(b) += c;
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
  mppi::critics::DynamicObstacleCritic,
  mppi::critics::CriticFunction)
