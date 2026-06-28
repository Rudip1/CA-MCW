// test/unit/critics/test_volumetric_critic.cpp — Phase 3.
//
// Validates that VolumetricCritic:
//   1. Produces zero contribution when no pointcloud has arrived yet (the
//      Phase-4-not-landed-yet graceful-degrade contract).
//   2. Penalises trajectories whose footprint cylinder intersects 3D points
//      and produces costs in the upstream magnitude band.
//   3. Trajectories far from any 3D obstacle get less cost than near ones.

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <nav2_mppi_controller/critic_data.hpp>
#include <nav2_mppi_controller/models/path.hpp>
#include <nav2_mppi_controller/models/state.hpp>
#include <nav2_mppi_controller/models/trajectories.hpp>
#include <nav2_mppi_controller/motion_models.hpp>
#include <nav2_mppi_controller/tools/parameters_handler.hpp>

#include "vf_robot_controller/critics/volumetric_critic.hpp"
#include "vf_robot_controller/controller/weight_cache.hpp"

namespace {

class TestVolumetricCritic : public mppi::critics::VolumetricCritic {
public:
  using mppi::critics::VolumetricCritic::initialize;
  using mppi::critics::VolumetricCritic::costmap_;
  using mppi::critics::VolumetricCritic::costmap_ros_;
  using mppi::critics::VolumetricCritic::parameters_handler_;
  using mppi::critics::VolumetricCritic::name_;
  using mppi::critics::VolumetricCritic::parent_name_;
  using mppi::critics::VolumetricCritic::parent_;
  using mppi::critics::VolumetricCritic::enabled_;
};

class VolumetricCriticTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!rclcpp::ok()) rclcpp::init(0, nullptr);
    node_ = std::make_shared<rclcpp_lifecycle::LifecycleNode>("volumetric_test");
    handler_ = std::make_unique<mppi::ParametersHandler>(node_);
    vf_robot_controller::WeightCache::instance().clear();
  }
  void TearDown() override {
    vf_robot_controller::WeightCache::instance().clear();
  }

  std::unique_ptr<TestVolumetricCritic> makeCritic() {
    auto c = std::make_unique<TestVolumetricCritic>();
    c->parameters_handler_ = handler_.get();
    c->name_ = "FollowPath.VolumetricCritic";
    c->parent_name_ = "FollowPath";
    c->parent_ = node_;
    c->enabled_ = true;
    c->initialize();
    return c;
  }

  std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node_;
  std::unique_ptr<mppi::ParametersHandler> handler_;
};

struct VolumetricFixture {
  mppi::models::State state;
  mppi::models::Trajectories trajectories;
  mppi::models::Path path;
  xt::xtensor<float, 1> costs;
  float model_dt;
  std::shared_ptr<mppi::MotionModel> motion_model;

  VolumetricFixture(size_t batch, size_t T, float lateral_offset) {
    trajectories.reset(batch, T);
    for (size_t b = 0; b < batch; ++b) {
      for (size_t t = 0; t < T; ++t) {
        trajectories.x(b, t) = 0.05f * static_cast<float>(t);
        trajectories.y(b, t) = lateral_offset;
      }
    }
    path.reset(2);
    path.x(0) = 0.0f; path.y(0) = 0.0f;
    path.x(1) = 1.0f; path.y(1) = 0.0f;
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

TEST_F(VolumetricCriticTest, ZeroCostWhenNoPointcloudPresent) {
  auto critic = makeCritic();
  VolumetricFixture fx(4, 20, 0.0f);
  auto data = fx.asCriticData();
  critic->score(data);
  for (size_t b = 0; b < fx.costs.size(); ++b) {
    EXPECT_FLOAT_EQ(fx.costs(b), 0.0f)
      << "VolumetricCritic must contribute zero when no /vf/voxel_filtered_pointcloud yet";
  }
}

TEST_F(VolumetricCriticTest, ProducesNonZeroCostsForTrajectoriesNearPoints) {
  auto critic = makeCritic();
  // Inject a dense cluster of points near x=0.5, y=0.0 at robot height.
  std::vector<std::array<float, 3>> pts;
  for (int i = 0; i < 200; ++i) {
    const float jitter = 0.001f * static_cast<float>(i);
    pts.push_back({0.5f + jitter, 0.0f, 0.2f});
  }
  critic->setPointsForTest(pts);

  VolumetricFixture fx(4, 20, 0.0f);
  auto data = fx.asCriticData();
  critic->score(data);

  float max_cost = 0.0f;
  for (size_t b = 0; b < fx.costs.size(); ++b) {
    max_cost = std::max(max_cost, fx.costs(b));
  }
  EXPECT_GT(max_cost, 100.0f)
    << "VolumetricCritic max cost too small to register against upstream critics";
}

TEST_F(VolumetricCriticTest, FarTrajectoryHasLowerCostThanNearTrajectory) {
  std::vector<std::array<float, 3>> pts;
  for (int i = 0; i < 200; ++i) {
    const float jitter = 0.001f * static_cast<float>(i);
    pts.push_back({0.5f + jitter, 0.0f, 0.2f});
  }

  auto near_critic = makeCritic();
  near_critic->setPointsForTest(pts);
  VolumetricFixture near(2, 20, 0.0f);
  auto near_data = near.asCriticData();
  near_critic->score(near_data);

  auto far_critic = makeCritic();
  far_critic->setPointsForTest(pts);
  VolumetricFixture far(2, 20, 3.0f);  // way off in y
  auto far_data = far.asCriticData();
  far_critic->score(far_data);

  EXPECT_GT(near.costs(0), far.costs(0))
    << "Trajectory passing through points must be costlier than one far away";
  EXPECT_GT(near.costs(0), 100.0f);
}

TEST_F(VolumetricCriticTest, HeightGateExcludesPointsAboveRobot) {
  std::vector<std::array<float, 3>> pts;
  for (int i = 0; i < 200; ++i) {
    // All points high above the robot top (default height_max == 0.40).
    pts.push_back({0.5f + 0.001f * i, 0.0f, 1.50f});
  }

  auto critic = makeCritic();
  critic->setPointsForTest(pts);
  VolumetricFixture fx(2, 20, 0.0f);
  auto data = fx.asCriticData();
  critic->score(data);
  for (size_t b = 0; b < fx.costs.size(); ++b) {
    EXPECT_FLOAT_EQ(fx.costs(b), 0.0f) << "Points above height_max must be ignored";
  }
}

}  // namespace
