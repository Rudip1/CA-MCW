// test/unit/perception/gcf/test_gcf_2d.cpp — Phase 4.

#include <gtest/gtest.h>

#include <memory>

#include <nav2_costmap_2d/cost_values.hpp>
#include <nav2_costmap_2d/costmap_2d.hpp>

#include "vf_robot_controller/perception/gcf/gcf_2d.hpp"

using vf_robot_controller::perception::gcf::Gcf2D;

namespace {

// Build a Costmap2D with the specified cells filled with `fill_value`.
std::shared_ptr<nav2_costmap_2d::Costmap2D> makeCostmap(
  unsigned int w, unsigned int h, double res, uint8_t default_val = 0)
{
  auto cm = std::make_shared<nav2_costmap_2d::Costmap2D>(
    w, h, res, /*origin_x=*/0.0, /*origin_y=*/0.0, default_val);
  return cm;
}

}  // namespace

TEST(Gcf2D, NoCostmapYieldsZero) {
  Gcf2D g(2.0);
  auto cell = g.query(0.0, 0.0);
  EXPECT_EQ(cell.complexity, 0.0);
}

TEST(Gcf2D, FreeCostmapYieldsZeroComplexity) {
  Gcf2D g(1.0);
  auto cm = makeCostmap(40, 40, 0.05, 0);
  g.setCostmap(cm);
  auto cell = g.query(1.0, 1.0);
  EXPECT_EQ(cell.complexity, 0.0);
  EXPECT_GT(cell.clearance_2d, 0.99);
  EXPECT_TRUE(cell.traversable);
}

TEST(Gcf2D, SaturatedInflationYieldsHighComplexity) {
  Gcf2D g(0.5);
  // Fill every cell with a high inflation value (cost 230) — no lethal.
  auto cm = makeCostmap(40, 40, 0.05, 230);
  g.setCostmap(cm);
  auto cell = g.query(1.0, 1.0);
  EXPECT_GT(cell.complexity, 0.95);
  EXPECT_LE(cell.complexity, 1.0);
  // No lethal cells → traversable.
  EXPECT_TRUE(cell.traversable);
}

TEST(Gcf2D, LethalNeighbourhoodMarksNonTraversable) {
  Gcf2D g(0.3);
  auto cm = makeCostmap(40, 40, 0.05, 0);
  // Stamp a lethal cell at (1, 1) world coords = cell (20, 20).
  cm->setCost(20, 20, nav2_costmap_2d::LETHAL_OBSTACLE);
  g.setCostmap(cm);
  auto cell = g.query(1.0, 1.0);
  EXPECT_FALSE(cell.traversable);
  EXPECT_GT(cell.complexity, 0.0);
}

TEST(Gcf2D, OutsideCostmapReturnsDefault) {
  Gcf2D g(0.3);
  auto cm = makeCostmap(40, 40, 0.05, 200);
  g.setCostmap(cm);
  // Costmap covers (0..2, 0..2). Query at (10, 10) is outside.
  auto cell = g.query(10.0, 10.0);
  EXPECT_EQ(cell.complexity, 0.0);
}
