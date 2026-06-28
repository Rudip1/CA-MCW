// nodes/context_node.cpp — 10 Hz navigation context classifier. Phase 5.
//
// Subscribes:
//   /vf/gcf_state    (std_msgs/Float32) from gcf_node
//   /odom            (nav_msgs/Odometry)
//   /goal_pose       (geometry_msgs/PoseStamped, optional, latches latest)
//
// Publishes:
//   /vf/context_state (std_msgs/Int8) — NavigationContext id (0..5, 255=UNKNOWN)
//
// Uses HysteresisClassifier from vf_perception_lib so the classification
// rules sit in a single, testable place. The node is a thin pump: cache
// inputs, build PerceptionState every tick, classify, publish.

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/int8.hpp>

#include "vf_robot_controller/perception/common/types.hpp"
#include "vf_robot_controller/perception/context/hysteresis_classifier.hpp"

using std::placeholders::_1;
namespace vfp = vf_robot_controller::perception;

class ContextNode : public rclcpp::Node {
public:
  ContextNode()
  : Node("context_node")
  {
    declare_parameter<double>("update_rate_hz", 10.0);
    declare_parameter<std::string>("gcf_topic", "/vf/gcf_state");
    declare_parameter<std::string>("odom_topic", "/odom");
    declare_parameter<std::string>("goal_topic", "/goal_pose");
    declare_parameter<std::string>("publish_topic", "/vf/context_state");
    declare_parameter<double>("approach_distance", 1.0);
    declare_parameter<double>("open_high", 0.20);
    declare_parameter<double>("open_low", 0.30);
    declare_parameter<double>("tight_low", 0.55);
    declare_parameter<double>("tight_high", 0.65);
    declare_parameter<double>("clutter_dynamic", 0.85);

    vfp::HysteresisThresholds t;
    t.approach_distance = static_cast<float>(get_parameter("approach_distance").as_double());
    t.open_high = static_cast<float>(get_parameter("open_high").as_double());
    t.open_low = static_cast<float>(get_parameter("open_low").as_double());
    t.tight_low = static_cast<float>(get_parameter("tight_low").as_double());
    t.tight_high = static_cast<float>(get_parameter("tight_high").as_double());
    t.clutter_dynamic = static_cast<float>(get_parameter("clutter_dynamic").as_double());
    classifier_.setThresholds(t);

    auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();

    gcf_sub_ = create_subscription<std_msgs::msg::Float32>(
      get_parameter("gcf_topic").as_string(), rclcpp::QoS(10),
      std::bind(&ContextNode::gcfCallback, this, _1));
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      get_parameter("odom_topic").as_string(), sensor_qos,
      std::bind(&ContextNode::odomCallback, this, _1));
    goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      get_parameter("goal_topic").as_string(), rclcpp::QoS(1),
      std::bind(&ContextNode::goalCallback, this, _1));

    pub_ = create_publisher<std_msgs::msg::Int8>(
      get_parameter("publish_topic").as_string(), rclcpp::QoS(10));

    const double rate_hz = get_parameter("update_rate_hz").as_double();
    const auto period = std::chrono::duration<double>(1.0 / std::max(0.1, rate_hz));
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&ContextNode::tick, this));

    RCLCPP_INFO(get_logger(),
      "context_node ready: rate=%.1fHz approach=%.2f thresholds(open %.2f/%.2f, tight %.2f/%.2f)",
      rate_hz, t.approach_distance, t.open_high, t.open_low, t.tight_low, t.tight_high);
  }

private:
  void gcfCallback(std_msgs::msg::Float32::SharedPtr msg) {
    last_gcf_.store(msg->data);
    has_gcf_.store(true);
  }
  void odomCallback(nav_msgs::msg::Odometry::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(state_mu_);
    rx_ = msg->pose.pose.position.x;
    ry_ = msg->pose.pose.position.y;
    has_odom_ = true;
  }
  void goalCallback(geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(state_mu_);
    gx_ = msg->pose.position.x;
    gy_ = msg->pose.position.y;
    has_goal_ = true;
  }

  void tick() {
    vfp::PerceptionState s;
    s.gcf_scalar = last_gcf_.load();
    s.gcf_fresh = has_gcf_.load();
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      if (!has_odom_) return;
      s.robot_pose.x = rx_;
      s.robot_pose.y = ry_;
      if (has_goal_) {
        const double dx = gx_ - rx_;
        const double dy = gy_ - ry_;
        s.distance_to_goal = static_cast<float>(std::sqrt(dx * dx + dy * dy));
      }
    }
    const auto ctx = classifier_.classify(s);
    std_msgs::msg::Int8 m;
    m.data = static_cast<int8_t>(ctx);
    pub_->publish(m);
  }

  vfp::HysteresisClassifier classifier_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr gcf_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Publisher<std_msgs::msg::Int8>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::atomic<float> last_gcf_{0.0f};
  std::atomic<bool> has_gcf_{false};
  std::mutex state_mu_;
  bool has_odom_{false};
  bool has_goal_{false};
  double rx_{0.0}, ry_{0.0};
  double gx_{0.0}, gy_{0.0};
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ContextNode>());
  rclcpp::shutdown();
  return 0;
}
