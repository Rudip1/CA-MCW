#include "vf_robot_controller/perception/pointcloud/voxel_filter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>

#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace vf_robot_controller::perception {

namespace {

struct VoxelKey {
  int32_t x, y, z;
  bool operator==(const VoxelKey & other) const noexcept
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash {
  size_t operator()(const VoxelKey & k) const noexcept
  {
    // Cantor-pairing-flavoured mix; 3 ints into 64-bit space.
    uint64_t h = static_cast<uint64_t>(k.x) * 73856093u;
    h ^= static_cast<uint64_t>(k.y) * 19349663u;
    h ^= static_cast<uint64_t>(k.z) * 83492791u;
    return static_cast<size_t>(h);
  }
};

struct Centroid {
  float x{0.0f}, y{0.0f}, z{0.0f};
  uint32_t n{0};
};

}  // namespace

VoxelFilter::VoxelFilter(float leaf_size)
: leaf_size_(leaf_size > 0.0f ? leaf_size : 0.05f)
{
}

size_t VoxelFilter::filter(
  const sensor_msgs::msg::PointCloud2 & in,
  std::vector<std::array<float, 3>> & out_points) const
{
  out_points.clear();
  if (in.data.empty() || in.width == 0 || in.height == 0) return 0;

  std::unordered_map<VoxelKey, Centroid, VoxelKeyHash> bins;
  // Modest overallocation reduces rehash cost on dense clouds.
  bins.reserve(static_cast<size_t>(in.width) * in.height / 32 + 64);

  const float inv_leaf = 1.0f / leaf_size_;

  sensor_msgs::PointCloud2ConstIterator<float> it_x(in, "x");
  sensor_msgs::PointCloud2ConstIterator<float> it_y(in, "y");
  sensor_msgs::PointCloud2ConstIterator<float> it_z(in, "z");

  const size_t n = static_cast<size_t>(in.width) * in.height;
  for (size_t i = 0; i < n; ++i, ++it_x, ++it_y, ++it_z) {
    const float x = *it_x, y = *it_y, z = *it_z;
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;

    VoxelKey key{
      static_cast<int32_t>(std::floor(x * inv_leaf)),
      static_cast<int32_t>(std::floor(y * inv_leaf)),
      static_cast<int32_t>(std::floor(z * inv_leaf))};

    auto & c = bins[key];
    c.x += x;
    c.y += y;
    c.z += z;
    c.n += 1;
  }

  out_points.reserve(bins.size());
  for (const auto & [key, c] : bins) {
    const float w = c.n > 0 ? 1.0f / static_cast<float>(c.n) : 1.0f;
    out_points.push_back({c.x * w, c.y * w, c.z * w});
  }
  return out_points.size();
}

void VoxelFilter::filter(
  const sensor_msgs::msg::PointCloud2 & in,
  sensor_msgs::msg::PointCloud2 & out) const
{
  std::vector<std::array<float, 3>> pts;
  filter(in, pts);

  out.header = in.header;
  out.height = 1;
  out.width = static_cast<uint32_t>(pts.size());
  out.is_bigendian = false;
  out.is_dense = true;

  sensor_msgs::PointCloud2Modifier mod(out);
  mod.setPointCloud2FieldsByString(1, "xyz");
  mod.resize(pts.size());

  if (pts.empty()) return;

  sensor_msgs::PointCloud2Iterator<float> ox(out, "x");
  sensor_msgs::PointCloud2Iterator<float> oy(out, "y");
  sensor_msgs::PointCloud2Iterator<float> oz(out, "z");
  for (const auto & p : pts) {
    *ox = p[0]; *oy = p[1]; *oz = p[2];
    ++ox; ++oy; ++oz;
  }
}

}  // namespace vf_robot_controller::perception
