// test/unit/perception/features/test_channel_slam_persistent.cpp — Phase 6.
//
// Validates SlamPersistentChannel layout, capability-flag wiring, and
// graceful zero-fill when the backend is missing or capability-restricted.
// Uses small in-test fake backends rather than the SQLite fixture so the
// test is isolated from disk state.

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "vf_robot_controller/perception/common/types.hpp"
#include "vf_robot_controller/perception/features/channels/channel_slam_persistent.hpp"
#include "vf_robot_controller/perception/map_backend/i_map_backend.hpp"

using vf_robot_controller::perception::BackendCapabilities;
using vf_robot_controller::perception::IMapBackend;
using vf_robot_controller::perception::PerceptionState;
using vf_robot_controller::perception::Pose2D;
using vf_robot_controller::perception::SlamPersistentChannel;
using vf_robot_controller::perception::StructuralFeatures3D;
using vf_robot_controller::perception::TopologicalFeatures;

namespace {

class FakeBackend : public IMapBackend {
public:
  bool available_{true};
  BackendCapabilities caps_{true, true, true};
  float persistent_value_{0.5f};
  std::optional<TopologicalFeatures> topo_;
  std::optional<StructuralFeatures3D> s3_;

  bool isAvailable() const override { return available_; }
  BackendCapabilities capabilities() const override { return caps_; }
  std::vector<float> queryPersistentObstacles(
    const Pose2D &, const std::vector<float> & angles,
    const std::vector<float> & radii) const override
  {
    return std::vector<float>(angles.size() * radii.size(), persistent_value_);
  }
  std::optional<TopologicalFeatures> queryTopology(const Pose2D &) const override
  { return topo_; }
  std::optional<StructuralFeatures3D> query3DStructure(const Pose2D &) const override
  { return s3_; }
};

}  // namespace

TEST(ChannelSlamPersistent, NameAndDim) {
  SlamPersistentChannel ch;
  EXPECT_EQ(ch.name(), "slam_persistent");
  EXPECT_EQ(ch.dim(), 40);
}

TEST(ChannelSlamPersistent, NullBackendZeroFills) {
  SlamPersistentChannel ch;
  PerceptionState s;  // map_backend == nullptr
  Eigen::VectorXf out = Eigen::VectorXf::Constant(40, -99.0f);
  ch.compute(s, out);
  EXPECT_TRUE(out.isZero());
}

TEST(ChannelSlamPersistent, FullBackendWritesAllBlocks) {
  auto backend = std::make_shared<FakeBackend>();
  backend->persistent_value_ = 0.4f;
  TopologicalFeatures t{};
  t.distance_to_loop_closure_ahead = 1.5f;
  t.distance_to_loop_closure_behind = 2.5f;
  t.keyframe_density_2m = 6.0f;
  t.distance_to_branch_point = 0.8f;
  t.visual_entropy = 1.2f;
  backend->topo_ = t;
  StructuralFeatures3D s3{};
  s3.ceiling_height = 2.4f;
  s3.floor_planarity = 0.92f;
  s3.vertical_clutter_robot_height = 0.3f;
  s3.vertical_clutter_head_height = 0.2f;
  s3.distinct_obstacle_clusters = 3;
  backend->s3_ = s3;

  SlamPersistentChannel ch;
  PerceptionState st;
  st.map_backend = backend;
  Eigen::VectorXf out(40);
  ch.compute(st, out);

  // Topology raw values (slots 0..4)
  EXPECT_FLOAT_EQ(out(0), 1.5f);
  EXPECT_FLOAT_EQ(out(1), 2.5f);
  EXPECT_FLOAT_EQ(out(2), 6.0f);
  EXPECT_FLOAT_EQ(out(4), 1.2f);

  // Persistent rosette: slots 12..27 should all be 0.4 (from fake).
  for (int i = 12; i < 28; ++i) {
    EXPECT_NEAR(out(i), 0.4f, 1e-5f);
  }

  // 3D structure raw values (slots 28..32)
  EXPECT_FLOAT_EQ(out(28), 2.4f);
  EXPECT_FLOAT_EQ(out(29), 0.92f);
  EXPECT_FLOAT_EQ(out(32), 3.0f);

  // Capability flags (slots 38, 39)
  EXPECT_FLOAT_EQ(out(38), 1.0f);
  EXPECT_FLOAT_EQ(out(39), 1.0f);
}

TEST(ChannelSlamPersistent, StaticBackendOnlyFillsPersistentBlock) {
  auto backend = std::make_shared<FakeBackend>();
  backend->caps_ = {true, false, false};  // mimic StaticMapBackend
  backend->persistent_value_ = 0.7f;
  // topo_/s3_ deliberately left empty — the channel must skip them.

  SlamPersistentChannel ch;
  ch.invalidateCacheForTest();
  PerceptionState st;
  st.map_backend = backend;
  Eigen::VectorXf out(40);
  ch.compute(st, out);

  // Topology block must be all zeros.
  for (int i = 0; i < 12; ++i) EXPECT_FLOAT_EQ(out(i), 0.0f);
  // Persistent block populated.
  for (int i = 12; i < 28; ++i) EXPECT_NEAR(out(i), 0.7f, 1e-5f);
  // 3D block all zeros.
  for (int i = 28; i < 38; ++i) EXPECT_FLOAT_EQ(out(i), 0.0f);
  // Capability flags reflect the partial caps.
  EXPECT_FLOAT_EQ(out(38), 0.0f);  // topology_ok
  EXPECT_FLOAT_EQ(out(39), 0.0f);  // structure_3d_ok
}

TEST(ChannelSlamPersistent, BackendNotReadyZeroFills) {
  auto backend = std::make_shared<FakeBackend>();
  backend->available_ = false;  // e.g. RTAB .db not yet found
  SlamPersistentChannel ch;
  ch.invalidateCacheForTest();
  PerceptionState st;
  st.map_backend = backend;
  Eigen::VectorXf out = Eigen::VectorXf::Constant(40, -1.0f);
  ch.compute(st, out);
  EXPECT_TRUE(out.isZero());
}

TEST(ChannelSlamPersistent, CapabilityFlagsAlwaysTwoSlotsAtEnd) {
  // No matter the backend, slots 38..39 are the capability flags and
  // never bleed into the rosette block. Verify by feeding a
  // 3D-only-no-topology backend.
  auto backend = std::make_shared<FakeBackend>();
  backend->caps_ = {true, false, true};
  StructuralFeatures3D s3{};
  s3.ceiling_height = 1.0f;
  backend->s3_ = s3;

  SlamPersistentChannel ch;
  ch.invalidateCacheForTest();
  PerceptionState st;
  st.map_backend = backend;
  Eigen::VectorXf out(40);
  ch.compute(st, out);
  EXPECT_FLOAT_EQ(out(38), 0.0f);
  EXPECT_FLOAT_EQ(out(39), 1.0f);
  EXPECT_FLOAT_EQ(out(28), 1.0f);
}
