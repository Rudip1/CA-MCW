// nodes/gcf_node.cpp — Decoupled GCF computation node. Phase 4.
//
// Architecture (per docs/architecture.md):
//   - 5 Hz timer.
//   - Subscribes (best-effort, low QoS) to: pointcloud(s), local costmap,
//     robot odometry.
//   - Voxel-filters the cloud at the subscriber callback.
//   - Computes GcfComposite at robot pose, publishes scalar [0,1] on
//     /vf/gcf_state every tick.
//   - Re-publishes the voxel-filtered cloud on
//     /vf/voxel_filtered_pointcloud for VolumetricCritic to consume.
//
// Critical invariants (the design notes):
//   - Never block the control loop. CorridorCritic / VolumetricCritic
//     subscribe to our outputs with their own staleness checks, so this
//     process can crash without breaking the controller.
//   - Voxel filter at the subscriber, not on the timer's hot path. Cuts
//     ~300k input points to ~5k.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/float32.hpp>
#include <tf2/exceptions.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <nav2_costmap_2d/costmap_2d.hpp>

#include "vf_robot_controller/perception/gcf/clutter_detector.hpp"
#include "vf_robot_controller/perception/gcf/gcf_2d.hpp"
#include "vf_robot_controller/perception/gcf/gcf_3d.hpp"
#include "vf_robot_controller/perception/gcf/gcf_composite.hpp"
#include "vf_robot_controller/perception/pointcloud/voxel_filter.hpp"

using std::placeholders::_1;
namespace vfp = vf_robot_controller::perception;
namespace vfg = vf_robot_controller::perception::gcf;

class GcfNode : public rclcpp::Node {
public:
  GcfNode()
  : Node("gcf_node"),
    voxel_filter_(0.05f)
  {
    // ── Parameters (matched to config/perception/perception.yaml) ──────
    declare_parameter<double>("update_rate_hz", 5.0);
    declare_parameter<double>("gcf_radius", 2.0);
    declare_parameter<double>("gcf_clutter_radius", 0.6);
    declare_parameter<double>("gcf_weight_2d", 0.4);
    declare_parameter<double>("gcf_weight_clutter", 0.3);
    declare_parameter<double>("gcf_weight_volumetric", 0.3);
    declare_parameter<bool>("use_3d", true);
    declare_parameter<double>("voxel_leaf_size", 0.05);
    declare_parameter<double>("height_min", 0.05);
    declare_parameter<double>("height_max", 1.5);
    declare_parameter<int>("saturation_count_3d", 50);
    declare_parameter<int>("saturation_count_clutter", 30);
    declare_parameter<std::vector<std::string>>(
      "pointcloud_topics",
      std::vector<std::string>{
        // Gazebo auto-prepends camera-link name to the ros namespace,
        // so the actual publishing topic is the verbose form
        // namespace + linkname + topicname. The shorter form exists
        // but is empty — confirmed via `ros2 topic list`.
        "/d435i/depth/d435i_depth/points",
        "/d455/depth/d455_depth/points"});
    declare_parameter<std::string>("costmap_topic", "/local_costmap/costmap");
    declare_parameter<std::string>("odom_topic", "/odom");
    declare_parameter<std::string>("publish_topic", "/vf/gcf_state");
    declare_parameter<std::string>(
      "voxel_publish_topic", "/vf/voxel_filtered_pointcloud");
    // Frame the published voxel cloud is expressed in. The frame must match
    // what MPPI's trajectories use — for VolumetricCritic the trajectory
    // points (tx, ty) come from `data.trajectories`, which Nav2 MPPI
    // populates in the local-costmap frame (`odom` per nav2_base.yaml).
    // Cloud points must therefore also be in `odom` so that the per-pose
    // 0.35 m radius check is comparing values in the same coordinate system.
    //
    // base_footprint would *seem* tempting (z = height above ground), but
    // it leaves (x, y) as robot-relative coords while trajectories are
    // world coords — the radius check then never matches and the critic
    // is silently zero. Use `odom`: z is still height above the floor
    // (since odom origin is at ground), and (x, y) align with trajectories.
    //
    // The transform is looked up once per source camera frame and cached
    // (cameras are rigidly mounted, so static).
    declare_parameter<std::string>("target_frame", "odom");
    target_frame_ = get_parameter("target_frame").as_string();
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    const double rate_hz = get_parameter("update_rate_hz").as_double();
    const double leaf = get_parameter("voxel_leaf_size").as_double();
    voxel_filter_.setLeafSize(static_cast<float>(leaf));

    // ── GCF components ────────────────────────────────────────────────
    gcf_2d_ = std::make_shared<vfg::Gcf2D>(get_parameter("gcf_radius").as_double());
    gcf_3d_ = std::make_shared<vfg::Gcf3D>(
      get_parameter("gcf_radius").as_double(),
      get_parameter("height_min").as_double(),
      get_parameter("height_max").as_double());
    gcf_3d_->setSaturationCount(
      static_cast<int>(get_parameter("saturation_count_3d").as_int()));

    clutter_ = std::make_shared<vfg::ClutterDetector>(
      get_parameter("gcf_clutter_radius").as_double());
    clutter_->setSaturationCount(
      static_cast<int>(get_parameter("saturation_count_clutter").as_int()));

    vfg::GcfCompositeWeights weights;
    weights.w_2d = get_parameter("gcf_weight_2d").as_double();
    weights.w_clutter = get_parameter("gcf_weight_clutter").as_double();
    weights.w_volumetric = get_parameter("use_3d").as_bool()
                             ? get_parameter("gcf_weight_volumetric").as_double()
                             : 0.0;
    composite_ = std::make_shared<vfg::GcfComposite>(
      gcf_2d_, gcf_3d_, clutter_, weights);

    // ── I/O ───────────────────────────────────────────────────────────
    auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();

    auto cloud_topics = get_parameter("pointcloud_topics").as_string_array();
    for (const auto & topic : cloud_topics) {
      auto sub = create_subscription<sensor_msgs::msg::PointCloud2>(
        topic, sensor_qos,
        [this, topic](sensor_msgs::msg::PointCloud2::SharedPtr msg) {
          this->cloudCallback(topic, std::move(msg));
        });
      cloud_subs_.push_back(sub);
      RCLCPP_INFO(get_logger(), "gcf_node: subscribed to %s", topic.c_str());
    }

    costmap_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      get_parameter("costmap_topic").as_string(),
      rclcpp::QoS(1).transient_local().reliable(),
      std::bind(&GcfNode::costmapCallback, this, _1));

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      get_parameter("odom_topic").as_string(), sensor_qos,
      std::bind(&GcfNode::odomCallback, this, _1));

    gcf_pub_ = create_publisher<std_msgs::msg::Float32>(
      get_parameter("publish_topic").as_string(), rclcpp::QoS(10));
    voxel_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      get_parameter("voxel_publish_topic").as_string(),
      rclcpp::QoS(1).best_effort());

    const auto period =
      std::chrono::duration<double>(1.0 / std::max(0.1, rate_hz));
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&GcfNode::tick, this));

    RCLCPP_INFO(
      get_logger(),
      "gcf_node ready: rate=%.1fHz radius=%.2f leaf=%.3f w=(2D %.2f / clutter %.2f / 3D %.2f)",
      rate_hz, get_parameter("gcf_radius").as_double(), leaf,
      weights.w_2d, weights.w_clutter, weights.w_volumetric);
  }

private:
  // Look up `target_frame_ ← src` at the cloud's timestamp. We do not
  // cache: target_frame_ defaults to `odom`, and `odom ← camera_frame`
  // = `odom ← base_link ← camera_link` includes the time-varying
  // `odom ← base_link` chain. Caching at t=0 would freeze the robot's
  // pose at startup. ~10 lookups/sec × <1ms each is negligible.
  // We log once when the first lookup succeeds so users can confirm
  // the chain is wired correctly.
  bool lookupTransformForCloud(
    const std::string & src,
    const rclcpp::Time & stamp,
    Eigen::Isometry3d & T)
  {
    geometry_msgs::msg::TransformStamped tfs;
    try {
      tfs = tf_buffer_->lookupTransform(
        target_frame_, src, stamp,
        tf2::durationFromSec(0.05));
    } catch (const tf2::TransformException & e) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "gcf_node: TF lookup %s -> %s not yet available (%s); dropping cloud",
        src.c_str(), target_frame_.c_str(), e.what());
      return false;
    }
    T = tf2::transformToEigen(tfs);
    {
      std::lock_guard<std::mutex> lock(tf_cache_mu_);
      if (tf_cache_.find(src) == tf_cache_.end()) {
        // First successful lookup for this source — log so users know
        // the chain is alive.
        RCLCPP_INFO(
          get_logger(),
          "gcf_node: TF chain %s -> %s now resolving",
          src.c_str(), target_frame_.c_str());
        tf_cache_[src] = T;
      }
    }
    return true;
  }

  void cloudCallback(
    const std::string & topic,
    sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (!msg) return;
    auto pts = std::make_shared<std::vector<std::array<float, 3>>>();
    voxel_filter_.filter(*msg, *pts);

    // Transform voxelized points from the camera optical frame into
    // target_frame_ (default `odom`). After this loop pts->{x,y} are
    // world (odom) coordinates, matching the frame MPPI trajectories
    // are integrated in (per nav2_base.yaml local_costmap.global_frame).
    // pts->z is height above the floor (odom origin sits at ground).
    Eigen::Isometry3d T;
    const rclcpp::Time stamp(msg->header.stamp);
    if (!lookupTransformForCloud(msg->header.frame_id, stamp, T)) {
      return;  // TF not ready; next cloud will retry.
    }
    for (auto & p : *pts) {
      const Eigen::Vector3d v(p[0], p[1], p[2]);
      const Eigen::Vector3d w = T * v;
      p = {
        static_cast<float>(w.x()),
        static_cast<float>(w.y()),
        static_cast<float>(w.z())};
    }

    {
      std::lock_guard<std::mutex> lock(cloud_mu_);
      latest_clouds_[topic] = pts;
      latest_cloud_header_ = msg->header;
      // Override frame_id — the points are now in target_frame_, even
      // though the original message header named the camera frame.
      latest_cloud_header_.frame_id = target_frame_;
    }
    has_cloud_.store(true);
  }

  void costmapCallback(nav_msgs::msg::OccupancyGrid::SharedPtr msg)
  {
    if (!msg) return;
    auto cm = std::make_shared<nav2_costmap_2d::Costmap2D>(*msg);
    gcf_2d_->setCostmap(cm);
    has_costmap_.store(true);
  }

  void odomCallback(nav_msgs::msg::Odometry::SharedPtr msg)
  {
    if (!msg) return;
    std::lock_guard<std::mutex> lock(pose_mu_);
    robot_x_ = msg->pose.pose.position.x;
    robot_y_ = msg->pose.pose.position.y;
    has_pose_.store(true);
  }

  void tick()
  {
    if (!has_pose_.load()) {
      // Nothing useful to publish until odom arrives.
      return;
    }

    // Merge all per-topic clouds into a single flat vector so Gcf3D and
    // ClutterDetector see one snapshot.
    auto merged = std::make_shared<std::vector<std::array<float, 3>>>();
    {
      std::lock_guard<std::mutex> lock(cloud_mu_);
      size_t total = 0;
      for (const auto & [t, pts] : latest_clouds_) {
        if (pts) total += pts->size();
      }
      merged->reserve(total);
      for (const auto & [t, pts] : latest_clouds_) {
        if (!pts) continue;
        merged->insert(merged->end(), pts->begin(), pts->end());
      }
    }
    gcf_3d_->setPoints(merged);
    clutter_->setPoints(merged);

    double rx, ry;
    {
      std::lock_guard<std::mutex> lock(pose_mu_);
      rx = robot_x_;
      ry = robot_y_;
    }

    auto cell = composite_->query(rx, ry);
    std_msgs::msg::Float32 msg;
    msg.data = static_cast<float>(cell.complexity);
    gcf_pub_->publish(msg);

    // Republish the merged voxel-filtered cloud so VolumetricCritic
    // doesn't have to subscribe to N raw camera topics.
    publishVoxelCloud(*merged);
  }

  void publishVoxelCloud(const std::vector<std::array<float, 3>> & pts)
  {
    if (pts.empty()) return;
    sensor_msgs::msg::PointCloud2 out;
    {
      std::lock_guard<std::mutex> lock(cloud_mu_);
      out.header = latest_cloud_header_;
    }
    if (out.header.frame_id.empty()) out.header.frame_id = "odom";
    out.header.stamp = now();
    out.height = 1;
    out.width = static_cast<uint32_t>(pts.size());
    out.is_bigendian = false;
    out.is_dense = true;

    sensor_msgs::PointCloud2Modifier mod(out);
    mod.setPointCloud2FieldsByString(1, "xyz");
    mod.resize(pts.size());
    sensor_msgs::PointCloud2Iterator<float> ox(out, "x");
    sensor_msgs::PointCloud2Iterator<float> oy(out, "y");
    sensor_msgs::PointCloud2Iterator<float> oz(out, "z");
    for (const auto & p : pts) {
      *ox = p[0]; *oy = p[1]; *oz = p[2];
      ++ox; ++oy; ++oz;
    }
    voxel_pub_->publish(out);
  }

  // GCF state.
  vfp::VoxelFilter voxel_filter_;
  std::shared_ptr<vfg::Gcf2D> gcf_2d_;
  std::shared_ptr<vfg::Gcf3D> gcf_3d_;
  std::shared_ptr<vfg::ClutterDetector> clutter_;
  std::shared_ptr<vfg::GcfComposite> composite_;

  // ROS I/O.
  std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> cloud_subs_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr gcf_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr voxel_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Cached state.
  std::map<std::string, std::shared_ptr<std::vector<std::array<float, 3>>>> latest_clouds_;
  std_msgs::msg::Header latest_cloud_header_;
  std::mutex cloud_mu_;
  double robot_x_{0.0}, robot_y_{0.0};
  std::mutex pose_mu_;
  std::atomic<bool> has_pose_{false};
  std::atomic<bool> has_cloud_{false};
  std::atomic<bool> has_costmap_{false};

  // TF: target frame for the published voxel cloud + cached per-source
  // static transforms. Cameras are rigidly mounted so each entry is
  // populated exactly once (on the first successful cloudCallback).
  std::string target_frame_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::map<std::string, Eigen::Isometry3d> tf_cache_;
  std::mutex tf_cache_mu_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GcfNode>());
  rclcpp::shutdown();
  return 0;
}
