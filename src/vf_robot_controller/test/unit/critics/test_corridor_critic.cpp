// test/unit/critics/test_corridor_critic.cpp — Phase 3.
//
// Validates that CorridorCritic:
//   1. Produces non-zero costs in the upstream magnitude band when given a
//      synthetic batch of trajectories that drift off the path.
//   2. Penalises off-path trajectories *more* than on-path ones.
//   3. The GCF multiplier scales the cost as expected (tight > open).
//
// We exercise the critic without a full Nav2 stack by constructing
// CriticData by hand and calling score() directly. The critic's
// configure-time subscribe is bypassed (no parent_ node here) so we use
// the setGcfForTest() seam to drive the multiplier.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <nav2_costmap_2d/costmap_2d_ros.hpp>
#include <nav2_mppi_controller/critic_data.hpp>
#include <nav2_mppi_controller/models/path.hpp>
#include <nav2_mppi_controller/models/state.hpp>
#include <nav2_mppi_controller/models/trajectories.hpp>
#include <nav2_mppi_controller/motion_models.hpp>
#include <nav2_mppi_controller/tools/parameters_handler.hpp>

#include "vf_robot_controller/critics/corridor_critic.hpp"
#include "vf_robot_controller/controller/weight_cache.hpp"

namespace {

// Helper: a friend-equivalent that exposes the protected configure entry
// without having to stand up a full lifecycle / CriticManager pipeline.
class TestCorridorCritic : public mppi::critics::CorridorCritic {
public:
  using mppi::critics::CorridorCritic::initialize;
  using mppi::critics::CorridorCritic::costmap_;
  using mppi::critics::CorridorCritic::costmap_ros_;
  using mppi::critics::CorridorCritic::parameters_handler_;
  using mppi::critics::CorridorCritic::name_;
  using mppi::critics::CorridorCritic::parent_name_;
  using mppi::critics::CorridorCritic::parent_;
  using mppi::critics::CorridorCritic::enabled_;
  using mppi::critics::CorridorCritic::weight_;
  using mppi::critics::CorridorCritic::yaml_weight_;
};

class CorridorCriticTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!rclcpp::ok()) rclcpp::init(0, nullptr);
    node_ = std::make_shared<rclcpp_lifecycle::LifecycleNode>("corridor_test");
    handler_ = std::make_unique<mppi::ParametersHandler>(node_);
    vf_robot_controller::WeightCache::instance().clear();
  }

  void TearDown() override {
    vf_robot_controller::WeightCache::instance().clear();
  }

  // Build a critic with a simple straight path along the +x axis.
  std::unique_ptr<TestCorridorCritic> makeCritic() {
    auto c = std::make_unique<TestCorridorCritic>();
    c->parameters_handler_ = handler_.get();
    c->name_ = "FollowPath.CorridorCritic";
    c->parent_name_ = "FollowPath";
    c->parent_ = node_;
    c->enabled_ = true;
    c->initialize();
    return c;
  }

  std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node_;
  std::unique_ptr<mppi::ParametersHandler> handler_;
};

// Build a CriticData with batch trajectories at a fixed lateral offset.
struct ScoringFixture {
  mppi::models::State state;
  mppi::models::Trajectories trajectories;
  mppi::models::Path path;
  xt::xtensor<float, 1> costs;
  float model_dt;
  std::shared_ptr<mppi::MotionModel> motion_model;

  ScoringFixture(size_t batch, size_t T, float lateral_offset, size_t n_path = 50) {
    trajectories.reset(batch, T);
    for (size_t b = 0; b < batch; ++b) {
      for (size_t t = 0; t < T; ++t) {
        trajectories.x(b, t) = 0.05f * static_cast<float>(t);
        trajectories.y(b, t) = lateral_offset;
      }
    }
    path.reset(n_path);
    for (size_t i = 0; i < n_path; ++i) {
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

TEST_F(CorridorCriticTest, ProducesNonZeroCostsForOffPathTrajectories) {
  auto critic = makeCritic();
  ScoringFixture fx(/*batch=*/8, /*T=*/20, /*lateral_offset=*/0.5f);

  auto data = fx.asCriticData();
  critic->score(data);

  // Expect every trajectory to have a non-zero cost (all drifted laterally
  // by 0.5m), and the magnitude to be in the upstream band.
  float max_cost = 0.0f, mean_cost = 0.0f;
  for (size_t b = 0; b < fx.costs.size(); ++b) {
    max_cost = std::max(max_cost, fx.costs(b));
    mean_cost += fx.costs(b);
  }
  mean_cost /= static_cast<float>(fx.costs.size());

  // Cost-magnitude rule (the design notes): per-trajectory contribution must live
  // in the same band as upstream critics (~1-10000), not 0-1.
  EXPECT_GT(max_cost, 100.0f) << "CorridorCritic cost too small — would be invisible in MPPI sum";
  EXPECT_LT(max_cost, 100000.0f) << "CorridorCritic cost too large — would dominate";

  // Sanity: max == mean since all trajectories are identical here.
  EXPECT_NEAR(max_cost, mean_cost, 1e-3f);
}

TEST_F(CorridorCriticTest, OnPathTrajectoryHasLowerCostThanOffPath) {
  auto critic = makeCritic();
  // On-path
  ScoringFixture on_path(4, 20, 0.0f);
  auto on_data = on_path.asCriticData();
  critic->score(on_data);

  // Reset weights so the multiplier doesn't carry between calls.
  vf_robot_controller::WeightCache::instance().clear();
  auto critic2 = makeCritic();
  ScoringFixture off_path(4, 20, 0.7f);
  auto off_data = off_path.asCriticData();
  critic2->score(off_data);

  EXPECT_LT(on_path.costs(0), off_path.costs(0))
    << "On-path trajectory should have lower cost than 0.7m off-path";
  EXPECT_GT(off_path.costs(0), 100.0f) << "Off-path cost must reach upstream band";
}

TEST_F(CorridorCriticTest, GcfMultiplierAmplifiesCostInTightCorridors) {
  // Open space: gcf == 0 → multiplier scaled to gcf_scale_min_ (default 0.5).
  auto critic_open = makeCritic();
  critic_open->setGcfForTest(0.0f);
  ScoringFixture open(4, 20, 0.5f);
  auto open_data = open.asCriticData();
  critic_open->score(open_data);

  // Tight space: gcf == 1 → multiplier scaled to gcf_scale_max_ (default 2.0).
  auto critic_tight = makeCritic();
  critic_tight->setGcfForTest(1.0f);
  ScoringFixture tight(4, 20, 0.5f);
  auto tight_data = tight.asCriticData();
  critic_tight->score(tight_data);

  EXPECT_GT(tight.costs(0), open.costs(0))
    << "Tight-corridor GCF should amplify the cost relative to open space";
  // Magnitude check: roughly 4x ratio (2.0/0.5)^power_, with power_=2 → 16x.
  // Allow wide tolerance because the cost computation isn't exactly square.
  EXPECT_GT(tight.costs(0) / open.costs(0), 2.0f);
}

TEST_F(CorridorCriticTest, MetaCriticMultiplierOverridesYamlWeight) {
  auto critic = makeCritic();

  // Baseline: no cache.
  ScoringFixture base(4, 20, 0.5f);
  auto base_data = base.asCriticData();
  critic->score(base_data);

  // With cache active and 0x multiplier, contribution should drop to zero.
  vf_robot_controller::WeightCache::instance().setActive(true);
  vf_robot_controller::WeightCache::instance().setMultiplier(
    "FollowPath.CorridorCritic", 0.0f);

  auto critic2 = makeCritic();
  ScoringFixture suppressed(4, 20, 0.5f);
  auto suppressed_data = suppressed.asCriticData();
  critic2->score(suppressed_data);

  EXPECT_GT(base.costs(0), 100.0f);
  EXPECT_LT(suppressed.costs(0), 1e-3f) << "Multiplier 0 should suppress the critic";
}

}  // namespace
