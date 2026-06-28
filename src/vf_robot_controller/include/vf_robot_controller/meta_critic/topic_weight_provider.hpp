// TopicWeightProvider — Phase 8 implementation.
//
// Subscribes to /vf_controller/meta_weights (std_msgs/Float32MultiArray) and
// returns the latest message as the weight vector. The publisher is the
// Python sidecar `metacritic_inference_node.py` (Phase 8 — runs torch or
// onnxruntime in Python and publishes per-cycle weights). The C++ controller
// reads them through this provider so the production system never depends on
// onnxruntime being installed.
//
// Staleness handling:
//   - If the latest message is older than `weight_timeout_ms`, return an
//     empty vector. The VFCriticManager treats empty as "fall back to
//     multiplier == 1.0" for every critic.
//   - The first failed lookup is logged at WARN; subsequent failures are
//     throttled to one per ~5 s.
//
// Thread safety: the subscriber callback runs on the controller's executor.
// We snapshot under a mutex; getWeights() reads under the same mutex.

#ifndef VF_ROBOT_CONTROLLER__META_CRITIC__TOPIC_WEIGHT_PROVIDER_HPP_
#define VF_ROBOT_CONTROLLER__META_CRITIC__TOPIC_WEIGHT_PROVIDER_HPP_

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include "vf_robot_controller/meta_critic/i_weight_provider.hpp"

namespace vf_robot_controller::meta_critic {

class TopicWeightProvider : public IWeightProvider {
public:
  TopicWeightProvider();

  // Phase 8 surface — equivalent to FixedWeightProvider::configure() but
  // also creates the subscription. Reads:
  //   <param_ns>.weight_topic            (default "/vf_controller/meta_weights")
  //   <param_ns>.weight_timeout_ms       (default 200)
  //   <param_ns>.fixed_weights           (used as fall-back when stale)
  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & node,
    const std::string & param_ns,
    int num_critics);

  int numCritics() const override;
  std::vector<float> getWeights(
    const Eigen::Ref<const Eigen::VectorXf> & features) override;
  std::string name() const override { return "topic"; }

  // Test seam — push a synthetic message into the cache.
  void injectMessageForTest(const std::vector<float> & weights);

private:
  void onWeightsMsg(const std_msgs::msg::Float32MultiArray::SharedPtr msg);

  int num_critics_{0};
  std::vector<float> fallback_weights_;
  std::vector<float> latest_weights_;
  rclcpp::Time latest_stamp_;
  std::chrono::milliseconds timeout_{200};

  rclcpp_lifecycle::LifecycleNode::WeakPtr node_weak_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_;
  rclcpp::Logger logger_{rclcpp::get_logger("TopicWeightProvider")};
  std::mutex mu_;
  bool warned_stale_once_{false};
};

}  // namespace vf_robot_controller::meta_critic

#endif  // VF_ROBOT_CONTROLLER__META_CRITIC__TOPIC_WEIGHT_PROVIDER_HPP_
