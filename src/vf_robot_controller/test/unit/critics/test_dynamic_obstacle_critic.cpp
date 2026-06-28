// test/unit/critics/test_dynamic_obstacle_critic.cpp — Phase 3.
//
// Validates that DynamicObstacleCritic:
//   1. Returns zero contribution when no costmap history exists yet
//      (graceful degradation; first cycle).
//   2. Returns zero when the snapshot history is identical (no obstacle
//      motion).
//   3. Produces non-zero costs in the upstream magnitude band when a new
//      obstacle appears in a trajectory's path between snapshots.
//   4. Honours the meta-critic WeightCache multiplier (zero suppresses).
//
// We bypass costmap_ entirely by seeding the snapshot history through the
// `seedHistoryForTest` test seam.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <nav2_mppi_controller/critic_data.hpp>
#include <nav2_mppi_controller/models/path.hpp>
#include <nav2_mppi_controller/models/state.hpp>
#include <nav2_mppi_controller/models/trajectories.hpp>
#include <nav2_mppi_controller/motion_models.hpp>
#include <nav2_mppi_controller/tools/parameters_handler.hpp>

#include "vf_robot_controller/critics/dynamic_obstacle_critic.hpp"
#include "vf_robot_controller/controller/weight_cache.hpp"

namespace {

class TestDynamicObstacleCritic : public mppi::critics::DynamicObstacleCritic {
public:
  using mppi::critics::DynamicObstacleCritic::initialize;
  using mppi::critics::DynamicObstacleCritic::costmap_;
  using mppi::critics::DynamicObstacleCritic::costmap_ros_;
  using mppi::critics::DynamicObstacleCritic::parameters_handler_;
  using mppi::critics::DynamicObstacleCritic::name_;
  using mppi::critics::DynamicObstacleCritic::parent_name_;
  using mppi::critics::DynamicObstacleCritic::parent_;
  using mppi::critics::DynamicObstacleCritic::enabled_;
  using mppi::critics::DynamicObstacleCritic::weight_;
  using mppi::critics::DynamicObstacleCritic::yaml_weight_;
};

class DynamicObstacleCriticTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!rclcpp::ok()) rclcpp::init(0, nullptr);
    node_ = std::make_shared<rclcpp_lifecycle::LifecycleNode>("dyn_obs_test");
    handler_ = std::make_unique<mppi::ParametersHandler>(node_);
    vf_robot_controller::WeightCache::instance().clear();
  }

  void TearDown() override {
    vf_robot_controller::WeightCache::instance().clear();
  }

  std::unique_ptr<TestDynamicObstacleCritic> makeCritic() {
    auto c = std::make_unique<TestDynamicObstacleCritic>();
    c->parameters_handler_ = handler_.get();
    c->name_ = "FollowPath.DynamicObstacleCritic";
    c->parent_name_ = "FollowPath";
    c->parent_ = node_;
    c->enabled_ = true;
    c->initialize();
    return c;
  }

  // Build a snapshot whose origin is at (-2, -2) and side 4m at 0.05m
  // resolution = 80x80 cells. Optionally stamp a high-cost cluster at
  // (obs_x, obs_y) of radius `r` cells.
  static mppi::critics::CostmapSnapshot makeSnapshot(
    bool with_obstacle, float obs_x = 0.0f, float obs_y = 0.0f,
    int r = 4, uint8_t value = 250)
  {
    mppi::critics::CostmapSnapshot s;
    s.origin_x = -2.0;
    s.origin_y = -2.0;
    s.resolution = 0.05;
    s.width = 80;
    s.height = 80;
    s.cells.assign(static_cast<size_t>(s.width) * s.height, 0);
    if (!with_obstacle) return s;

    const int ci = static_cast<int>((obs_x - s.origin_x) / s.resolution);
    const int cj = static_cast<int>((obs_y - s.origin_y) / s.resolution);
    for (int dj = -r; dj <= r; ++dj) {
      for (int di = -r; di <= r; ++di) {
        const int i = ci + di;
        const int j = cj + dj;
        if (i < 0 || j < 0 ||
            i >= static_cast<int>(s.width) ||
            j >= static_cast<int>(s.height)) continue;
        if (di * di + dj * dj <= r * r) {
          s.cells[j * s.width + i] = value;
        }
      }
    }
    return s;
  }

  std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node_;
  std::unique_ptr<mppi::ParametersHandler> handler_;
};

// Trajectories travelling along +x from x=0 through obstacle region.
struct Fixture {
  mppi::models::State state;
  mppi::models::Trajectories trajectories;
  mppi::models::Path path;
  xt::xtensor<float, 1> costs;
  float model_dt;
  std::shared_ptr<mppi::MotionModel> motion_model;

  Fixture(size_t batch, size_t T) {
    trajectories.reset(batch, T);
    for (size_t b = 0; b < batch; ++b) {
      for (size_t t = 0; t < T; ++t) {
        // Each batch travels along +x, with batch index controlling lateral
        // offset so we can place some clear of an obstacle.
        trajectories.x(b, t) = 0.05f * static_cast<float>(t);
        trajectories.y(b, t) = static_cast<float>(b) * 0.05f - 0.1f;
      }
    }
    path.reset(20);
    for (size_t i = 0; i < 20; ++i) {
      path.x(i) = 0.05f * static_cast<float>(i);
      path.y(i) = 0.0f;
    }
    costs = xt::zeros<float>({batch});
    model_dt = 0.05f;
    motion_model = std::make_shared<mppi::DiffDriveMotionModel>();
    state.reset(static_cast<unsigned int>(batch), static_cast<unsigned int>(T));
  }

  mppi::CriticData asCriticData() {
    return mppi::CriticData{
      state, trajectories, path, costs, model_dt,
      false, nullptr, motion_model, std::nullopt, std::nullopt};
  }
};

TEST_F(DynamicObstacleCriticTest, ZeroCostWhenHistoryEmpty) {
  auto critic = makeCritic();
  Fixture fx(/*batch=*/4, /*T=*/20);
  auto data = fx.asCriticData();

  // No costmap, no seeded history → score() should bail with zeros.
  critic->score(data);
  for (size_t b = 0; b < fx.costs.size(); ++b) {
    EXPECT_FLOAT_EQ(fx.costs(b), 0.0f);
  }
}

TEST_F(DynamicObstacleCriticTest, ZeroCostWhenSnapshotsAreIdentical) {
  auto critic = makeCritic();
  // Two identical snapshots without any obstacle motion.
  std::deque<mppi::critics::CostmapSnapshot> hist;
  hist.push_back(makeSnapshot(/*with_obstacle=*/false));
  hist.push_back(makeSnapshot(/*with_obstacle=*/false));
  critic->seedHistoryForTest(std::move(hist));

  Fixture fx(/*batch=*/4, /*T=*/20);
  auto data = fx.asCriticData();
  critic->score(data);
  for (size_t b = 0; b < fx.costs.size(); ++b) {
    EXPECT_FLOAT_EQ(fx.costs(b), 0.0f);
  }
}

TEST_F(DynamicObstacleCriticTest, ProducesCostInUpstreamBandWhenObstacleAppears) {
  auto critic = makeCritic();

  // Old snapshot: clean. New snapshot: obstacle at (0.5, 0.0) — directly on
  // the trajectory of batch index 2 (lateral offset = 2*0.05 - 0.1 = 0.0m).
  std::deque<mppi::critics::CostmapSnapshot> hist;
  hist.push_back(makeSnapshot(false));
  hist.push_back(makeSnapshot(true, /*obs_x=*/0.5f, /*obs_y=*/0.0f));
  critic->seedHistoryForTest(std::move(hist));

  Fixture fx(/*batch=*/5, /*T=*/40);
  auto data = fx.asCriticData();
  critic->score(data);

  float max_cost = 0.0f;
  for (size_t b = 0; b < fx.costs.size(); ++b) {
    max_cost = std::max(max_cost, fx.costs(b));
  }
  // Cost-magnitude rule: per-trajectory contribution must reach the
  // upstream band. With ~250 delta values × time decay × default weight 0.4
  // summed over multiple sampled poses, expect well into the hundreds.
  EXPECT_GT(max_cost, 100.0f)
    << "DynamicObstacleCritic cost too small — would be invisible in MPPI sum";
  EXPECT_LT(max_cost, 100000.0f);

  // Trajectory passing through the obstacle (batch 2) should cost more
  // than trajectories that miss it laterally (batch 0, batch 4 are 0.1m
  // and 0.1m away).
  EXPECT_GT(fx.costs(2), fx.costs(0));
  EXPECT_GT(fx.costs(2), fx.costs(4));
}

TEST_F(DynamicObstacleCriticTest, MetaCriticMultiplierSuppressesContribution) {
  auto critic = makeCritic();
  std::deque<mppi::critics::CostmapSnapshot> hist;
  hist.push_back(makeSnapshot(false));
  hist.push_back(makeSnapshot(true, 0.5f, 0.0f));
  critic->seedHistoryForTest(std::move(hist));

  // Baseline cost.
  Fixture base(5, 40);
  auto base_data = base.asCriticData();
  critic->score(base_data);

  // Now zero out via WeightCache multiplier and re-score with a fresh critic.
  vf_robot_controller::WeightCache::instance().setActive(true);
  vf_robot_controller::WeightCache::instance().setMultiplier(
    "FollowPath.DynamicObstacleCritic", 0.0f);

  auto critic2 = makeCritic();
  std::deque<mppi::critics::CostmapSnapshot> hist2;
  hist2.push_back(makeSnapshot(false));
  hist2.push_back(makeSnapshot(true, 0.5f, 0.0f));
  critic2->seedHistoryForTest(std::move(hist2));

  Fixture suppressed(5, 40);
  auto suppressed_data = suppressed.asCriticData();
  critic2->score(suppressed_data);

  EXPECT_GT(base.costs(2), 100.0f);
  EXPECT_LT(suppressed.costs(2), 1e-3f)
    << "Multiplier 0 should suppress DynamicObstacleCritic to zero";
}

}  // namespace
