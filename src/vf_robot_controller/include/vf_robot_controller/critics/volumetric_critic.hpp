// VolumetricCritic — VF custom critic for nav2_mppi_controller. Phase 3.
//
// Penalises trajectories that pass through 3D obstacles invisible to the 2D
// costmap (low-hanging shelves, table edges, chair seats above scan height).
// Subscribes to /vf/voxel_filtered_pointcloud (sensor_msgs/PointCloud2,
// frame: odom). For each predicted pose along each candidate trajectory,
// counts pointcloud hits inside a footprint cylinder (radius +
// height-band).
//
// **Phase 4 dependency.** The proper voxel-filtered pointcloud topic does
// not exist yet — Phase 4 will publish it from gcf_node. Until then, this
// critic subscribes to the topic with no upstream publisher. With no data,
// it produces zero contribution and the controller behaves as if the
// critic weren't present (graceful degrade).
//
// **No PCL dependency.** We don't deserialise the pointcloud through pcl
// types — sensor_msgs::PointCloud2 already has the raw float32 buffer; we
// stride into it with a simple iterator and a per-cycle subsample. Phase 4
// will replace this with the proper PCL voxel filter shared with gcf_node.
//
// **Cost magnitude.** Per-trajectory cost = mean_hits_per_pose × weight.
// With a moderately cluttered scene (dozens of points within 0.5 m of a
// pose) and weight default 5.0, cost lands in [0, 200] — same band as
// CostCritic.
//
// **Plugin namespace.** mppi::critics so upstream's CriticManager finds it.

#ifndef VF_ROBOT_CONTROLLER__CRITICS__VOLUMETRIC_CRITIC_HPP_
#define VF_ROBOT_CONTROLLER__CRITICS__VOLUMETRIC_CRITIC_HPP_

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "nav2_mppi_controller/critic_function.hpp"

namespace mppi::critics {

class VolumetricCritic : public CriticFunction {
public:
  VolumetricCritic() = default;
  ~VolumetricCritic() override = default;

  void initialize() override;
  void score(CriticData & data) override;

  // Test seam: inject a flat list of (x, y, z) points for the cached cloud.
  void setPointsForTest(std::vector<std::array<float, 3>> pts) {
    std::lock_guard<std::mutex> lock(cloud_mu_);
    cached_points_ = std::move(pts);
    has_cloud_.store(true);
    if (clock_) {
      cloud_stamp_ = clock_->now();
    }
  }

protected:
  // YAML-tunable parameters.
  unsigned int power_{1};
  float weight_{5.0f};
  float yaml_weight_{5.0f};
  float footprint_radius_{0.35f};   // metres; cylinder radius around each pose.
  float height_min_{0.05f};          // metres; below the floor margin.
  float height_max_{0.40f};          // metres; below robot top.
  int trajectory_point_step_{4};
  int point_subsample_stride_{4};   // Every Nth point retained at subscriber.
  double cloud_stale_seconds_{1.0};
  size_t max_points_{8000};          // Hard cap after subsample.

  // Subscriber state — pure cache.
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  std::vector<std::array<float, 3>> cached_points_;
  rclcpp::Time cloud_stamp_;
  std::atomic<bool> has_cloud_{false};
  std::mutex cloud_mu_;

  std::shared_ptr<rclcpp::Clock> clock_;

  bool isCloudFresh();
};

}  // namespace mppi::critics

#endif  // VF_ROBOT_CONTROLLER__CRITICS__VOLUMETRIC_CRITIC_HPP_
