// VoxelFilter — hash-bin voxel grid downsampler. Phase 4.
//
// Why hand-rolled instead of pcl::VoxelGrid:
//   PCL's CMake config on Ubuntu 22.04 drags in a broken QHULL imported
//   target (see CMakeLists.txt comment). Phase 3's VolumetricCritic
//   already strides directly into PointCloud2's float32 buffer without
//   PCL — we keep that pattern.
//
// Algorithm:
//   1. Bin every point into a 3D integer cell (floor(x / leaf), ...).
//   2. Keep one centroid per occupied bin (running mean).
//   3. Output the centroids as either a flat float vector or a fresh
//      PointCloud2.
//
// Performance: at 0.05 m leaf size on a 5x5x2 m volume that's 200x200x40
// = 1.6M possible bins, but typical RealSense clouds populate <5%. With
// std::unordered_map and a good hash this comfortably handles 300k input
// points in well under 5 ms.

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__POINTCLOUD__VOXEL_FILTER_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__POINTCLOUD__VOXEL_FILTER_HPP_

#include <array>
#include <cstdint>
#include <vector>

#include <sensor_msgs/msg/point_cloud2.hpp>

namespace vf_robot_controller::perception {

class VoxelFilter {
public:
  explicit VoxelFilter(float leaf_size = 0.05f);

  void setLeafSize(float leaf) { leaf_size_ = leaf; }
  float leafSize() const { return leaf_size_; }

  // Filter `in` into `out_points` as a flat vector of (x, y, z) triples.
  // Skips NaN/Inf points. Empty input → empty output.
  // Returns the number of output points.
  size_t filter(
    const sensor_msgs::msg::PointCloud2 & in,
    std::vector<std::array<float, 3>> & out_points) const;

  // Same filter, but output as a PointCloud2 with the input's header.
  // Frame is preserved.
  void filter(
    const sensor_msgs::msg::PointCloud2 & in,
    sensor_msgs::msg::PointCloud2 & out) const;

private:
  float leaf_size_;
};

}  // namespace vf_robot_controller::perception

#endif  // VF_ROBOT_CONTROLLER__PERCEPTION__POINTCLOUD__VOXEL_FILTER_HPP_
