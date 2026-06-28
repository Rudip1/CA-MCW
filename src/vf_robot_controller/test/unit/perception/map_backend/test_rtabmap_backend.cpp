// test/unit/perception/map_backend/test_rtabmap_backend.cpp — Phase 6.
//
// Validates that RtabmapBackend opens the SQLite fixture in WAL read-only
// mode, reads keyframe pose blobs, counts loop closures, and answers all
// three query types. Also covers the missing-file fallback path and the
// busy-DB tolerance via a second concurrent reader.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <thread>
#include <vector>

#include "vf_robot_controller/perception/map_backend/rtabmap_backend.hpp"

using vf_robot_controller::perception::Pose2D;
using vf_robot_controller::perception::RtabmapBackend;

namespace {

std::string fixtureDb()
{
  std::filesystem::path p = std::filesystem::current_path();
  for (int i = 0; i < 10; ++i) {
    auto c = p / "src" / "vf_robot_controller" / "test" / "fixtures" /
             "sample_rtabmap.db";
    if (std::filesystem::exists(c)) return c.string();
    if (!p.has_parent_path() || p.parent_path() == p) break;
    p = p.parent_path();
  }
  if (const char * home = std::getenv("HOME")) {
    std::filesystem::path c = std::filesystem::path(home) / "CA-MCW" /
      "src" / "vf_robot_controller" / "test" / "fixtures" /
      "sample_rtabmap.db";
    if (std::filesystem::exists(c)) return c.string();
  }
  return "";
}

}  // namespace

TEST(RtabmapBackend, OpensFixtureAndReadsKeyframes) {
  const std::string db = fixtureDb();
  ASSERT_FALSE(db.empty()) << "fixture not present";
  RtabmapBackend bk(db);
  EXPECT_TRUE(bk.isAvailable());
  EXPECT_EQ(bk.keyframeCountForTest(), 5u);
  EXPECT_EQ(bk.loopClosureCountForTest(), 1u);
  const auto c = bk.capabilities();
  EXPECT_TRUE(c.persistent_2d);
  EXPECT_TRUE(c.topology);
  EXPECT_TRUE(c.structure_3d);
}

TEST(RtabmapBackend, MissingFileReportsUnavailable) {
  RtabmapBackend bk("/this/file/does/not/exist.db");
  EXPECT_FALSE(bk.isAvailable());
  EXPECT_EQ(bk.keyframeCountForTest(), 0u);
  // Queries on unavailable backend return zero-filled output / nullopt.
  auto out = bk.queryPersistentObstacles({0, 0, 0}, {0.0f}, {1.0f});
  EXPECT_EQ(out.size(), 1u);
  EXPECT_FLOAT_EQ(out[0], 0.0f);
  EXPECT_FALSE(bk.queryTopology({0, 0, 0}).has_value());
  EXPECT_FALSE(bk.query3DStructure({0, 0, 0}).has_value());
}

TEST(RtabmapBackend, TopologyReportsKeyframeDensity) {
  const std::string db = fixtureDb();
  ASSERT_FALSE(db.empty());
  RtabmapBackend bk(db);
  ASSERT_TRUE(bk.isAvailable());
  // Robot at (2, 0): keyframes at x=0..4 spaced 1 m apart along y=0.
  // Within 2 m: ids 1..5 -> all 5 keyframes are within sqrt(2^2)=2 m.
  // (Distances: 2, 1, 0, 1, 2 — all <= 2 m exactly.)
  auto t = bk.queryTopology({2.0, 0.0, 0.0});
  ASSERT_TRUE(t.has_value());
  EXPECT_GE(t->keyframe_density_2m, 1.0f);
  // Visual entropy ~ log1p(loop_closure_count=1) ~= log(2) ~= 0.693.
  EXPECT_GT(t->visual_entropy, 0.0f);
  EXPECT_LT(t->visual_entropy, 5.0f);
  // Distance to branch point (= nearest keyframe distance) is small.
  EXPECT_LT(t->distance_to_branch_point, 1.0f);
}

TEST(RtabmapBackend, PersistentObstaclesAreInZeroOneRange) {
  const std::string db = fixtureDb();
  ASSERT_FALSE(db.empty());
  RtabmapBackend bk(db);
  ASSERT_TRUE(bk.isAvailable());
  // Sample 4 angles at radius 1 m around the start of the trajectory.
  std::vector<float> angles = {0.0f, 1.5708f, 3.14159f, -1.5708f};
  std::vector<float> radii  = {1.0f, 2.0f};
  auto out = bk.queryPersistentObstacles({0.0, 0.0, 0.0}, angles, radii);
  ASSERT_EQ(out.size(), 8u);
  for (float v : out) {
    EXPECT_GE(v, 0.0f);
    EXPECT_LE(v, 1.0f);
  }
}

TEST(RtabmapBackend, ConcurrentReadersDoNotInterfere) {
  // Two backends pointing at the same .db with the WAL+busy_timeout open
  // path should succeed (no SQLITE_BUSY thrown to caller).
  const std::string db = fixtureDb();
  ASSERT_FALSE(db.empty());
  std::vector<std::thread> threads;
  std::atomic<int> ok{0};
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&]() {
      RtabmapBackend bk(db);
      if (bk.isAvailable() && bk.keyframeCountForTest() == 5u) ++ok;
    });
  }
  for (auto & t : threads) t.join();
  EXPECT_EQ(ok.load(), 4);
}

TEST(RtabmapBackend, Structure3DBoundedRanges) {
  const std::string db = fixtureDb();
  ASSERT_FALSE(db.empty());
  RtabmapBackend bk(db);
  ASSERT_TRUE(bk.isAvailable());
  auto s = bk.query3DStructure({2.0, 0.0, 0.0});
  ASSERT_TRUE(s.has_value());
  // Fixture keyframes all sit at z=0.5, so ceiling=0.5, planarity≈1.
  EXPECT_NEAR(s->ceiling_height, 0.5f, 1e-3f);
  EXPECT_GE(s->floor_planarity, 0.0f);
  EXPECT_LE(s->floor_planarity, 1.0f);
}
