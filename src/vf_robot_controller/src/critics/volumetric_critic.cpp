// VolumetricCritic — Phase 3.
// See header for the design rationale.

#include "vf_robot_controller/critics/volumetric_critic.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "vf_robot_controller/controller/weight_cache.hpp"

namespace mppi::critics {

namespace {

// Decode a PointCloud2 stride-by-stride without pulling in PCL. Phase 4
// will replace this with the shared voxel filter, but for Phase 3 we want
// to subscribe to whatever someone is publishing and survive the fact
// that the upstream voxel filter does not yet exist.
//
// Returns true when both x and y offsets were resolved. z is optional and
// defaulted to 0 if absent (some 2D synthetic clouds omit z).
bool getXYZOffsets(
  const sensor_msgs::msg::PointCloud2 & msg,
  uint32_t & ox, uint32_t & oy, uint32_t & oz, bool & has_z)
{
  bool have_x = false, have_y = false;
  has_z = false;
  for (const auto & f : msg.fields) {
    if (f.name == "x") { ox = f.offset; have_x = true; }
    else if (f.name == "y") { oy = f.offset; have_y = true; }
    else if (f.name == "z") { oz = f.offset; has_z = true; }
  }
  return have_x && have_y;
}

}  // namespace

void VolumetricCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(power_, "cost_power", 1);
  getParam(weight_, "cost_weight", 5.0f);
  getParam(footprint_radius_, "footprint_radius", 0.35f);
  getParam(height_min_, "height_min", 0.05f);
  getParam(height_max_, "height_max", 0.40f);
  getParam(trajectory_point_step_, "trajectory_point_step", 4);
  getParam(point_subsample_stride_, "point_subsample_stride", 4);
  getParam(cloud_stale_seconds_, "cloud_stale_seconds", 1.0);
  // max_points_ is size_t; fetch as int and assign.
  int mp = static_cast<int>(max_points_);
  getParam(mp, "max_points", static_cast<int>(max_points_));
  if (mp > 0) max_points_ = static_cast<size_t>(mp);

  yaml_weight_ = weight_;

  if (auto node = parent_.lock()) {
    clock_ = node->get_clock();
    cloud_stamp_ = clock_->now() - rclcpp::Duration::from_seconds(1e6);

    auto qos = rclcpp::QoS(1).best_effort();
    cloud_sub_ = node->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/vf/voxel_filtered_pointcloud", qos,
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
        uint32_t ox = 0, oy = 0, oz = 0;
        bool has_z = false;
        if (!getXYZOffsets(*msg, ox, oy, oz, has_z)) {
          return;
        }
        const size_t step = msg->point_step;
        const size_t n = msg->width * msg->height;
        const int stride = std::max(1, point_subsample_stride_);
        std::vector<std::array<float, 3>> pts;
        pts.reserve(std::min<size_t>(max_points_, n / stride + 1));
        for (size_t i = 0; i < n && pts.size() < max_points_; i += stride) {
          const uint8_t * base = msg->data.data() + i * step;
          float x = 0.0f, y = 0.0f, z = 0.0f;
          std::memcpy(&x, base + ox, sizeof(float));
          std::memcpy(&y, base + oy, sizeof(float));
          if (has_z) std::memcpy(&z, base + oz, sizeof(float));
          pts.push_back({x, y, z});
        }

        std::lock_guard<std::mutex> lock(cloud_mu_);
        cached_points_ = std::move(pts);
        cloud_stamp_ = clock_->now();
        has_cloud_.store(true);
      });
  }

  RCLCPP_INFO(
    logger_,
    "VolumetricCritic initialised: power=%u weight=%.2f r=%.2f h=[%.2f,%.2f] "
    "stride=%d max_points=%zu",
    power_, weight_, footprint_radius_, height_min_, height_max_,
    point_subsample_stride_, max_points_);
}

bool VolumetricCritic::isCloudFresh()
{
  if (!has_cloud_.load() || !clock_) return false;
  rclcpp::Time stamp;
  {
    std::lock_guard<std::mutex> lock(cloud_mu_);
    stamp = cloud_stamp_;
  }
  const auto age = (clock_->now() - stamp).seconds();
  return age < cloud_stale_seconds_;
}

void VolumetricCritic::score(CriticData & data)
{
  auto & cache = ::vf_robot_controller::WeightCache::instance();
  if (cache.isActive()) {
    auto m = cache.getMultiplier(getName());
    weight_ = yaml_weight_ * (m.has_value() ? *m : 1.0f);
  }

  const bool collect = cache.isCostCollectionActive();
  std::vector<float> before;
  if (collect) {
    before.assign(data.costs.cbegin(), data.costs.cend());
  }

  // Snapshot the points under the lock, then operate on the local copy.
  // This bounds the scope of contention with the subscriber callback.
  std::vector<std::array<float, 3>> points;
  if (enabled_ && isCloudFresh()) {
    std::lock_guard<std::mutex> lock(cloud_mu_);
    points = cached_points_;
  }

  if (points.empty()) {
    if (collect) {
      cache.recordDelta(getName(), std::vector<float>(data.costs.size(), 0.0f));
    }
    return;
  }

  const size_t batch = data.trajectories.x.shape()[0];
  const size_t T = data.trajectories.x.shape()[1];
  const int step = std::max(1, trajectory_point_step_);
  const float r2 = footprint_radius_ * footprint_radius_;

  std::vector<float> cost(batch, 0.0f);

  // Pre-filter points by height once — saves work in the per-trajectory loop.
  std::vector<std::pair<float, float>> xy_hits;
  xy_hits.reserve(points.size());
  for (const auto & p : points) {
    if (p[2] >= height_min_ && p[2] <= height_max_) {
      xy_hits.emplace_back(p[0], p[1]);
    }
  }

  if (xy_hits.empty()) {
    if (collect) {
      cache.recordDelta(getName(), std::vector<float>(data.costs.size(), 0.0f));
    }
    return;
  }

  for (size_t b = 0; b < batch; ++b) {
    float sum_hits = 0.0f;
    unsigned int n_poses = 0;
    for (size_t t = 0; t < T; t += static_cast<size_t>(step)) {
      const float tx = data.trajectories.x(b, t);
      const float ty = data.trajectories.y(b, t);
      int hits = 0;
      for (const auto & xy : xy_hits) {
        const float dx = xy.first - tx;
        const float dy = xy.second - ty;
        if (dx * dx + dy * dy <= r2) ++hits;
      }
      sum_hits += static_cast<float>(hits);
      ++n_poses;
    }
    if (n_poses > 0) cost[b] = sum_hits / static_cast<float>(n_poses);
  }

  for (size_t b = 0; b < batch; ++b) {
    float c = cost[b] * weight_;
    if (power_ > 1u) {
      c = std::pow(c, static_cast<int>(power_));
    }
    data.costs(b) += c;
  }

  if (collect) {
    std::vector<float> delta(data.costs.size());
    for (size_t i = 0; i < delta.size(); ++i) {
      delta[i] = data.costs(i) - before[i];
    }
    cache.recordDelta(getName(), std::move(delta));
  }
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(
  mppi::critics::VolumetricCritic,
  mppi::critics::CriticFunction)
