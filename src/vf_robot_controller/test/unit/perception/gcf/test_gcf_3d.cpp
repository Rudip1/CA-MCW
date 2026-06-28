// test/unit/perception/gcf/test_gcf_3d.cpp — Phase 4.

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <vector>

#include "vf_robot_controller/perception/gcf/clutter_detector.hpp"
#include "vf_robot_controller/perception/gcf/gcf_3d.hpp"

using vf_robot_controller::perception::gcf::ClutterDetector;
using vf_robot_controller::perception::gcf::Gcf3D;

TEST(Gcf3D, NoPointsYieldsZero) {
  Gcf3D g;
  auto cell = g.query(0.0, 0.0);
  EXPECT_EQ(cell.complexity, 0.0);
}

TEST(Gcf3D, PointsInRadiusYieldHighComplexity) {
  Gcf3D g(0.5, 0.0, 2.0);
  g.setSaturationCount(5);
  auto pts = std::make_shared<std::vector<std::array<float, 3>>>();
  for (int i = 0; i < 10; ++i) {
    pts->push_back({0.1f * i, 0.0f, 0.5f});  // all within 0.5 m of origin in x
  }
  g.setPoints(pts);
  auto cell = g.query(0.0, 0.0);
  EXPECT_GT(cell.complexity, 0.5);
}

TEST(Gcf3D, HeightGateExcludesOutsideBand) {
  Gcf3D g(1.0, 0.5, 1.0);
  g.setSaturationCount(5);
  auto pts = std::make_shared<std::vector<std::array<float, 3>>>();
  // All points well above the gate.
  for (int i = 0; i < 100; ++i) {
    pts->push_back({0.0f, 0.0f, 5.0f});
  }
  g.setPoints(pts);
  auto cell = g.query(0.0, 0.0);
  EXPECT_EQ(cell.complexity, 0.0);
}

TEST(Gcf3D, PointsBeyondRadiusIgnored) {
  Gcf3D g(0.3, 0.0, 2.0);
  g.setSaturationCount(5);
  auto pts = std::make_shared<std::vector<std::array<float, 3>>>();
  for (int i = 0; i < 100; ++i) {
    pts->push_back({5.0f, 0.0f, 0.5f});  // 5 m away from query
  }
  g.setPoints(pts);
  auto cell = g.query(0.0, 0.0);
  EXPECT_EQ(cell.complexity, 0.0);
}

TEST(ClutterDetector, IgnoresHeight) {
  ClutterDetector d(0.5);
  d.setSaturationCount(5);
  auto pts = std::make_shared<std::vector<std::array<float, 3>>>();
  // Mix of heights, all within 0.5 m horizontal.
  pts->push_back({0.1f, 0.0f, 0.0f});
  pts->push_back({0.0f, 0.1f, 1.5f});
  pts->push_back({0.2f, 0.0f, -0.5f});
  pts->push_back({0.0f, 0.3f, 3.0f});
  pts->push_back({0.4f, 0.0f, 0.5f});
  d.setPoints(pts);
  auto cell = d.query(0.0, 0.0);
  EXPECT_FLOAT_EQ(cell.complexity, 1.0f);
  EXPECT_FLOAT_EQ(cell.clutter_density, 1.0f);
}

TEST(ClutterDetector, EmptyCloudYieldsZero) {
  ClutterDetector d(0.5);
  auto cell = d.query(0.0, 0.0);
  EXPECT_EQ(cell.complexity, 0.0);
  EXPECT_EQ(cell.clutter_density, 0.0);
}
