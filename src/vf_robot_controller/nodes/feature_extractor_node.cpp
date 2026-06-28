// nodes/feature_extractor_node.cpp — 20 Hz feature vector publisher. Phase 5.
//
// Subscribes to all upstream perception topics, builds a PerceptionState
// snapshot every tick, runs FeatureExtractor, publishes a flat
// std_msgs/Float32MultiArray on /vf/features.
//
// The MultiArray layout encodes channel boundaries: dim[0] is "feature"
// with size totalDim, dim[1] is "channels" with stride/size per channel
// — so subscribers can recover per-channel slices.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int8.hpp>
#include <vf_robot_messages/msg/mppi_critics_stats.hpp>

#include <nav2_costmap_2d/costmap_2d.hpp>

#include "vf_robot_controller/perception/common/types.hpp"
#include "vf_robot_controller/perception/features/feature_extractor.hpp"
#include "vf_robot_controller/perception/map_backend/i_map_backend.hpp"
#include "vf_robot_controller/perception/map_backend/rtabmap_backend.hpp"
#include "vf_robot_controller/perception/map_backend/static_map_backend.hpp"
#include "vf_robot_controller/perception/map_backend/cuvslam_backend.hpp"

using std::placeholders::_1;
namespace vfp = vf_robot_controller::perception;

class FeatureExtractorNode : public rclcpp::Node {
public:
  FeatureExtractorNode()
  : Node("feature_extractor_node")
  {
    declare_parameter<double>("update_rate_hz", 20.0);
    declare_parameter<std::string>("publish_topic", "/vf/features");
    declare_parameter<std::vector<std::string>>(
      "enabled_channels",
      std::vector<std::string>{"robot_state", "context", "path_geometry",
                                "gcf_rosette", "critic_history",
                                "obstacle_dynamics"});
    declare_parameter<std::string>("odom_topic", "/odom");
    declare_parameter<std::string>("gcf_topic", "/vf/gcf_state");
    declare_parameter<std::string>("context_topic", "/vf/context_state");
    declare_parameter<std::string>("plan_topic", "/plan");
    declare_parameter<std::string>("goal_topic", "/goal_pose");
    declare_parameter<std::string>("costmap_topic", "/local_costmap/costmap");
    declare_parameter<std::string>("critic_costs_topic", "/vf/per_critic_costs");
    declare_parameter<int>("history_capacity", 10);
    // Phase 6: persistent-map backend selection (in-process, no service hop).
    declare_parameter<std::string>("map_backend", "none");  // none|rtabmap|static|cuvslam
    declare_parameter<std::string>("rtabmap_db_path", "");
    declare_parameter<std::string>("static_map_yaml", "");

    history_capacity_ = std::max(1, static_cast<int>(get_parameter("history_capacity").as_int()));

    // Instantiate the selected backend. Failures are logged but never
    // propagated — a missing backend leaves slam_persistent zero-filled.
    map_backend_ = makeMapBackend();

    auto channel_names = get_parameter("enabled_channels").as_string_array();
    for (const auto & name : channel_names) {
      auto ch = vfp::makeChannel(name);
      if (!ch) {
        RCLCPP_WARN(get_logger(), "Unknown channel '%s' — skipped", name.c_str());
        continue;
      }
      RCLCPP_INFO(get_logger(), "Channel registered: %s (%d dims)",
                  name.c_str(), ch->dim());
      extractor_.addChannel(std::move(ch));
    }
    RCLCPP_INFO(get_logger(), "FeatureExtractor total dim: %d", extractor_.totalDim());

    auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      get_parameter("odom_topic").as_string(), sensor_qos,
      std::bind(&FeatureExtractorNode::odomCallback, this, _1));
    gcf_sub_ = create_subscription<std_msgs::msg::Float32>(
      get_parameter("gcf_topic").as_string(), rclcpp::QoS(10),
      std::bind(&FeatureExtractorNode::gcfCallback, this, _1));
    context_sub_ = create_subscription<std_msgs::msg::Int8>(
      get_parameter("context_topic").as_string(), rclcpp::QoS(10),
      std::bind(&FeatureExtractorNode::contextCallback, this, _1));
    plan_sub_ = create_subscription<nav_msgs::msg::Path>(
      get_parameter("plan_topic").as_string(), rclcpp::QoS(1),
      std::bind(&FeatureExtractorNode::planCallback, this, _1));
    goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      get_parameter("goal_topic").as_string(), rclcpp::QoS(1),
      std::bind(&FeatureExtractorNode::goalCallback, this, _1));
    costmap_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      get_parameter("costmap_topic").as_string(),
      rclcpp::QoS(1).transient_local().reliable(),
      std::bind(&FeatureExtractorNode::costmapCallback, this, _1));
    critic_sub_ = create_subscription<vf_robot_messages::msg::MppiCriticsStats>(
      get_parameter("critic_costs_topic").as_string(), rclcpp::QoS(20),
      std::bind(&FeatureExtractorNode::criticCallback, this, _1));

    pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
      get_parameter("publish_topic").as_string(), rclcpp::QoS(10));

    const double rate_hz = get_parameter("update_rate_hz").as_double();
    const auto period = std::chrono::duration<double>(1.0 / std::max(0.1, rate_hz));
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&FeatureExtractorNode::tick, this));

    RCLCPP_INFO(get_logger(),
      "feature_extractor_node ready: rate=%.1fHz channels=%zu total_dim=%d",
      rate_hz, channel_names.size(), extractor_.totalDim());
  }

private:
  void odomCallback(nav_msgs::msg::Odometry::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(mu_);
    state_.robot_pose.x = msg->pose.pose.position.x;
    state_.robot_pose.y = msg->pose.pose.position.y;
    // Manual yaw from quaternion — avoids the tf2_geometry_msgs link dep.
    {
      const auto & q = msg->pose.pose.orientation;
      const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
      const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
      state_.robot_pose.theta = std::atan2(siny_cosp, cosy_cosp);
    }
    state_.velocity.x() = static_cast<float>(msg->twist.twist.linear.x);
    state_.velocity.y() = static_cast<float>(msg->twist.twist.linear.y);
    state_.velocity.z() = static_cast<float>(msg->twist.twist.angular.z);
    has_odom_ = true;
  }
  void gcfCallback(std_msgs::msg::Float32::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(mu_);
    state_.gcf_scalar = msg->data;
    state_.gcf_fresh = true;
  }
  void contextCallback(std_msgs::msg::Int8::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(mu_);
    state_.context_id = static_cast<uint8_t>(msg->data);
  }
  void planCallback(nav_msgs::msg::Path::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(mu_);
    state_.path.clear();
    state_.path.reserve(msg->poses.size());
    for (const auto & p : msg->poses) {
      state_.path.push_back({static_cast<float>(p.pose.position.x),
                             static_cast<float>(p.pose.position.y)});
    }
  }
  void goalCallback(geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(mu_);
    goal_x_ = msg->pose.position.x;
    goal_y_ = msg->pose.position.y;
    has_goal_ = true;
  }
  void costmapCallback(nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    auto cm = std::make_shared<nav2_costmap_2d::Costmap2D>(*msg);
    std::lock_guard<std::mutex> lock(mu_);
    state_.costmap_prev = state_.costmap_now;
    state_.costmap_now = cm;
  }
  void criticCallback(vf_robot_messages::msg::MppiCriticsStats::SharedPtr msg) {
    vfp::CriticCostSample s;
    s.costs = msg->costs_sum;  // float32[]
    std::lock_guard<std::mutex> lock(mu_);
    state_.critic_history.push_back(std::move(s));
    while (static_cast<int>(state_.critic_history.size()) > history_capacity_) {
      state_.critic_history.erase(state_.critic_history.begin());
    }
  }

  void tick() {
    vfp::PerceptionState snap;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (!has_odom_) return;
      snap = state_;
      if (has_goal_) {
        const double dx = goal_x_ - state_.robot_pose.x;
        const double dy = goal_y_ - state_.robot_pose.y;
        snap.distance_to_goal = static_cast<float>(std::sqrt(dx * dx + dy * dy));
      }
    }
    snap.map_backend = map_backend_;

    auto vec = extractor_.extract(snap);

    std_msgs::msg::Float32MultiArray msg;
    msg.layout.data_offset = 0;
    {
      std_msgs::msg::MultiArrayDimension feature_dim;
      feature_dim.label = "feature";
      feature_dim.size = static_cast<uint32_t>(vec.size());
      feature_dim.stride = static_cast<uint32_t>(vec.size());
      msg.layout.dim.push_back(feature_dim);
    }
    msg.data.assign(vec.data(), vec.data() + vec.size());
    pub_->publish(msg);
  }

  std::shared_ptr<vfp::IMapBackend> makeMapBackend()
  {
    const std::string sel = get_parameter("map_backend").as_string();
    if (sel == "rtabmap") {
      const auto path = get_parameter("rtabmap_db_path").as_string();
      auto b = std::make_shared<vfp::RtabmapBackend>(path);
      if (b->isAvailable()) {
        RCLCPP_INFO(get_logger(), "RtabmapBackend ready: %s", path.c_str());
        return b;
      }
      RCLCPP_WARN(get_logger(),
        "RtabmapBackend unavailable (db: '%s'); trying static fallback",
        path.c_str());
      // Graceful fallback: if a static map yaml was provided we degrade to
      // 2D-only persistent-obstacle features. Topology + 3D zero-fill.
      const auto static_path = get_parameter("static_map_yaml").as_string();
      if (!static_path.empty()) {
        auto s = std::make_shared<vfp::StaticMapBackend>(static_path);
        if (s->isAvailable()) {
          RCLCPP_INFO(get_logger(), "Falling back to StaticMapBackend: %s",
            static_path.c_str());
          return s;
        }
      }
      RCLCPP_WARN(get_logger(),
        "No persistent-map backend available; slam_persistent channel will zero-fill");
      return nullptr;
    }
    if (sel == "static") {
      const auto static_path = get_parameter("static_map_yaml").as_string();
      auto s = std::make_shared<vfp::StaticMapBackend>(static_path);
      if (s->isAvailable()) {
        RCLCPP_INFO(get_logger(), "StaticMapBackend ready: %s", static_path.c_str());
        return s;
      }
      RCLCPP_WARN(get_logger(),
        "StaticMapBackend unavailable (yaml: '%s'); slam_persistent will zero-fill",
        static_path.c_str());
      return nullptr;
    }
    if (sel == "cuvslam") {
      RCLCPP_WARN(get_logger(),
        "CuvslamBackend selected but is a Phase 11 stub; slam_persistent will zero-fill");
      return std::make_shared<vfp::CuvslamBackend>();
    }
    if (sel != "none" && !sel.empty()) {
      RCLCPP_WARN(get_logger(),
        "Unknown map_backend '%s'; slam_persistent will zero-fill", sel.c_str());
    }
    return nullptr;
  }

  vfp::FeatureExtractor extractor_;
  vfp::PerceptionState state_;
  int history_capacity_{10};
  std::shared_ptr<vfp::IMapBackend> map_backend_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr gcf_sub_;
  rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr context_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr plan_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_sub_;
  rclcpp::Subscription<vf_robot_messages::msg::MppiCriticsStats>::SharedPtr critic_sub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::mutex mu_;
  bool has_odom_{false};
  bool has_goal_{false};
  double goal_x_{0.0}, goal_y_{0.0};
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FeatureExtractorNode>());
  rclcpp::shutdown();
  return 0;
}
