// OnnxWeightProvider — Phase 8 implementation.
//
// Loads an ONNX file at configure() time, subscribes to /vf/features so it
// has the latest input vector, and runs forward inference inside getWeights()
// to produce per-critic weight multipliers.
//
// Build-time onnxruntime detection
// --------------------------------
// We do NOT hard-depend on onnxruntime at the package level — many target
// systems (developer laptops, the apt CI image we use) do not ship the
// onnxruntime C++ library. CMake probes for onnxruntime via find_path /
// find_library and defines `VF_HAS_ONNXRUNTIME` when found. When the macro
// is NOT defined:
//   * The class still compiles; the inference call returns an empty vector.
//   * VFCriticManager treats empty as "use multiplier == 1.0 for every
//     critic", which is functionally identical to FixedWeightProvider.
//   * A loud one-time WARN tells the operator to install onnxruntime or use
//     the topic-based provider (Python sidecar) instead.
//
// When onnxruntime IS present, the implementation runs the model in-process
// — no topic round-trip. This is the production deployment path.
//
// Either way, the class honours the IWeightProvider contract exactly:
// getWeights() never blocks, never throws.

#ifndef VF_ROBOT_CONTROLLER__META_CRITIC__ONNX_WEIGHT_PROVIDER_HPP_
#define VF_ROBOT_CONTROLLER__META_CRITIC__ONNX_WEIGHT_PROVIDER_HPP_

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include "vf_robot_controller/meta_critic/i_weight_provider.hpp"

namespace vf_robot_controller::meta_critic {

class OnnxWeightProvider : public IWeightProvider {
public:
  OnnxWeightProvider();
  ~OnnxWeightProvider() override;

  // Reads:
  //   <param_ns>.onnx_path                (default "")
  //   <param_ns>.norm_path                (default "")
  //   <param_ns>.expected_in_dim          (default 170)
  //   <param_ns>.weight_timeout_ms        (default 200)
  //   <param_ns>.features_topic           (default "/vf/features")
  //   <param_ns>.fixed_weights            (used as fall-back)
  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & node,
    const std::string & param_ns,
    int num_critics);

  int numCritics() const override;
  std::vector<float> getWeights(
    const Eigen::Ref<const Eigen::VectorXf> & features) override;
  std::string name() const override { return "onnx"; }

  // True when the underlying ONNX session loaded successfully. False when
  // onnxruntime is missing at build time, the .onnx path is empty, or load
  // failed at runtime — in all cases getWeights() returns the fallback.
  bool isModelLoaded() const;

private:
  void onFeaturesMsg(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
  std::vector<float> runForward(const std::vector<float> & x);  // noexcept-ish
  void loadNorm(const std::string & path);
  void normalize(std::vector<float> & x) const;

  int num_critics_{0};
  int expected_in_dim_{170};
  std::vector<float> fallback_weights_;
  std::string onnx_path_;

  std::vector<float> latest_features_;
  rclcpp::Time latest_stamp_;
  std::chrono::milliseconds timeout_{200};

  std::vector<float> norm_mean_;
  std::vector<float> norm_std_;

  rclcpp_lifecycle::LifecycleNode::WeakPtr node_weak_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_;
  rclcpp::Logger logger_{rclcpp::get_logger("OnnxWeightProvider")};
  std::mutex mu_;
  bool warned_stale_once_{false};
  bool warned_no_runtime_once_{false};

  // Opaque pimpl pointer for the onnxruntime Session — lives in the .cpp so
  // headers don't depend on onnxruntime at compile time.
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vf_robot_controller::meta_critic

#endif  // VF_ROBOT_CONTROLLER__META_CRITIC__ONNX_WEIGHT_PROVIDER_HPP_
