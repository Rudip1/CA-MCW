// test/unit/perception/map_backend/test_static_map_backend.cpp — Phase 6.
//
// Validates that StaticMapBackend opens the fixture pgm/yaml, reports
// availability + capabilities, and answers raycast queries with sane
// values (free interior -> 0, near-wall direction -> > 0, off-map -> 1).

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <string>

#include "vf_robot_controller/perception/map_backend/static_map_backend.hpp"

using vf_robot_controller::perception::Pose2D;
using vf_robot_controller::perception::StaticMapBackend;

namespace {

// Locate the fixture relative to the source tree. CMake doesn't pass the
// path in (vf_add_gtest is generic), so we resolve it from the binary's
// build directory walking up to src/.
std::string fixturePath()
{
  // Walk from current working directory up looking for src/vf_robot_controller.
  std::filesystem::path p = std::filesystem::current_path();
  for (int i = 0; i < 10; ++i) {
    auto candidate = p / "src" / "vf_robot_controller" / "test" / "fixtures"
                     / "sample_static_map" / "map.yaml";
    if (std::filesystem::exists(candidate)) return candidate.string();
    if (!p.has_parent_path() || p.parent_path() == p) break;
    p = p.parent_path();
  }
  // Fallback: COLCON_PREFIX_PATH and well-known dev location.
  if (const char * home = std::getenv("HOME")) {
    std::filesystem::path c = std::filesystem::path(home) / "CA-MCW" /
      "src" / "vf_robot_controller" / "test" / "fixtures" /
      "sample_static_map" / "map.yaml";
    if (std::filesystem::exists(c)) return c.string();
  }
  return "";
}

}  // namespace

TEST(StaticMapBackend, OpensFixtureAndReportsCapabilities) {
  const std::string p = fixturePath();
  ASSERT_FALSE(p.empty()) << "fixture not found";
  StaticMapBackend bk(p);
  EXPECT_TRUE(bk.isAvailable());
  const auto c = bk.capabilities();
  EXPECT_TRUE(c.persistent_2d);
  EXPECT_FALSE(c.topology);
  EXPECT_FALSE(c.structure_3d);
}

TEST(StaticMapBackend, MissingFileReportsUnavailable) {
  StaticMapBackend bk("/nonexistent/path/to/map.yaml");
  EXPECT_FALSE(bk.isAvailable());
  // Queries on an unavailable backend must not crash and must return
  // length angles*radii zero-filled output.
  std::vector<float> out = bk.queryPersistentObstacles(
    {0.0, 0.0, 0.0}, {0.0f, 1.5708f}, {1.0f, 2.0f});
  EXPECT_EQ(out.size(), 4u);
  for (float v : out) EXPECT_FLOAT_EQ(v, 0.0f);
}

TEST(StaticMapBackend, TopologyAnd3DReturnNullopt) {
  const std::string p = fixturePath();
  ASSERT_FALSE(p.empty());
  StaticMapBackend bk(p);
  ASSERT_TRUE(bk.isAvailable());
  EXPECT_FALSE(bk.queryTopology({0.0, 0.0, 0.0}).has_value());
  EXPECT_FALSE(bk.query3DStructure({0.0, 0.0, 0.0}).has_value());
}

TEST(StaticMapBackend, RaycastDetectsWallInOneDirection) {
  const std::string p = fixturePath();
  ASSERT_FALSE(p.empty());
  StaticMapBackend bk(p);
  ASSERT_TRUE(bk.isAvailable());
  // Fixture: 20x20 grid, 0.1 m resolution, origin (-1, -1).
  // Wall row is at j=10 (world y from -1 + 10*0.1 = 0.0 to 0.1) for
  // i in [5..15) -> world x in [-0.5, 0.5). So from (0,0) facing +y the
  // ray hits the wall almost immediately; +x sees free interior; +y
  // direction returns near 1 (very close hit).
  Pose2D robot{0.0, -0.5, 0.0};  // 0.5 m below the wall, looking up
  const std::vector<float> angles = {1.5708f};  // +y
  const std::vector<float> radii = {1.0f};
  auto out = bk.queryPersistentObstacles(robot, angles, radii);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_GT(out[0], 0.2f);  // hit somewhere within 1 m

  // Same pose, looking -y (away from wall): no hit, output 0 (or off-map = 1
  // if the ray exits the grid). With radius 0.4 m we stay on-map.
  Pose2D robot2{0.0, -0.5, 0.0};
  auto out2 = bk.queryPersistentObstacles(robot2, {-1.5708f}, {0.4f});
  EXPECT_FLOAT_EQ(out2[0], 0.0f);
}

TEST(StaticMapBackend, OffMapIsTreatedAsBlocked) {
  const std::string p = fixturePath();
  ASSERT_FALSE(p.empty());
  StaticMapBackend bk(p);
  ASSERT_TRUE(bk.isAvailable());
  // Robot well outside the 2 m x 2 m map. Any ray hits "off-map" almost
  // immediately and reports nearly 1.0.
  Pose2D far_robot{10.0, 10.0, 0.0};
  auto out = bk.queryPersistentObstacles(far_robot, {0.0f}, {1.0f});
  EXPECT_GT(out[0], 0.5f);
}
