// test/unit/perception/test_ring_buffer.cpp — Phase 5.

#include <gtest/gtest.h>

#include "vf_robot_controller/perception/common/ring_buffer.hpp"

using vf_robot_controller::perception::RingBuffer;

TEST(RingBuffer, EmptyOnConstruction) {
  RingBuffer<int> rb(4);
  EXPECT_EQ(rb.size(), 0u);
  EXPECT_EQ(rb.capacity(), 4u);
  EXPECT_TRUE(rb.empty());
  EXPECT_FALSE(rb.full());
}

TEST(RingBuffer, PushFillsToCapacity) {
  RingBuffer<int> rb(3);
  rb.push(1);
  EXPECT_EQ(rb.size(), 1u);
  EXPECT_EQ(rb.latest(), 1);
  EXPECT_EQ(rb.oldest(), 1);
  rb.push(2);
  rb.push(3);
  EXPECT_TRUE(rb.full());
  EXPECT_EQ(rb.oldest(), 1);
  EXPECT_EQ(rb.latest(), 3);
}

TEST(RingBuffer, EvictsOldestWhenOverCapacity) {
  RingBuffer<int> rb(3);
  rb.push(1); rb.push(2); rb.push(3); rb.push(4); rb.push(5);
  EXPECT_EQ(rb.size(), 3u);
  EXPECT_EQ(rb.oldest(), 3);
  EXPECT_EQ(rb.latest(), 5);
  EXPECT_EQ(rb.at(0), 3);
  EXPECT_EQ(rb.at(1), 4);
  EXPECT_EQ(rb.at(2), 5);
}

TEST(RingBuffer, ClearResets) {
  RingBuffer<int> rb(2);
  rb.push(1); rb.push(2);
  rb.clear();
  EXPECT_EQ(rb.size(), 0u);
  EXPECT_TRUE(rb.empty());
}

TEST(RingBuffer, OutOfRangeThrows) {
  RingBuffer<int> rb(3);
  rb.push(1);
  EXPECT_THROW(rb.at(5), std::out_of_range);
}

TEST(RingBuffer, ZeroCapacityClampedToOne) {
  RingBuffer<int> rb(0);
  EXPECT_EQ(rb.capacity(), 1u);
  rb.push(42);
  EXPECT_EQ(rb.latest(), 42);
}
