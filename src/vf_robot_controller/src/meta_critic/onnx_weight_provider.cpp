// OnnxWeightProvider — Phase 8.
//
// CMake passes -DVF_HAS_ONNXRUNTIME when it can find onnxruntime headers and
// the .so on the system. Without it the provider falls back to "always
// return fallback_weights" — the controller behaves exactly like a
// FixedWeightProvider would.

#include "vf_robot_controller/meta_critic/onnx_weight_provider.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>

#include <nav2_util/node_utils.hpp>
#include <rclcpp/qos.hpp>

#ifdef VF_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace vf_robot_controller::meta_critic {

struct OnnxWeightProvider::Impl
{
#ifdef VF_HAS_ONNXRUNTIME
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "OnnxWeightProvider"};
  std::unique_ptr<Ort::Session> session;
  std::string in_name;
  std::string out_name;
  std::vector<int64_t> in_shape;  // (1, D)
  bool ready{false};
#else
  bool ready{false};
#endif
};

OnnxWeightProvider::OnnxWeightProvider()
  : impl_(std::make_unique<Impl>()) {}

OnnxWeightProvider::~OnnxWeightProvider() = default;

void OnnxWeightProvider::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & node_weak,
  const std::string & param_ns,
  int num_critics)
{
  num_critics_ = num_critics;
  node_weak_ = node_weak;
  fallback_weights_.assign(num_critics, 1.0f);

  auto node = node_weak.lock();
  if (!node) {
    return;
  }
  logger_ = node->get_logger();

  // ── Parameters ───────────────────────────────────────────────────────────
  std::string features_topic = "/vf/features";
  std::string norm_path;
  int timeout_ms = 200;
  std::vector<double> fallback;
  nav2_util::declare_parameter_if_not_declared(
    node, param_ns + ".onnx_path", rclcpp::ParameterValue(std::string("")));
  nav2_util::declare_parameter_if_not_declared(
    node, param_ns + ".norm_path", rclcpp::ParameterValue(std::string("")));
  nav2_util::declare_parameter_if_not_declared(
    node, param_ns + ".expected_in_dim", rclcpp::ParameterValue(170));
  nav2_util::declare_parameter_if_not_declared(
    node, param_ns + ".weight_timeout_ms",
    rclcpp::ParameterValue(timeout_ms));
  nav2_util::declare_parameter_if_not_declared(
    node, param_ns + ".features_topic",
    rclcpp::ParameterValue(features_topic));
  nav2_util::declare_parameter_if_not_declared(
    node, param_ns + ".fixed_weights",
    rclcpp::ParameterValue(std::vector<double>{}));

  node->get_parameter(param_ns + ".onnx_path", onnx_path_);
  node->get_parameter(param_ns + ".norm_path", norm_path);
  node->get_parameter(param_ns + ".expected_in_dim", expected_in_dim_);
  node->get_parameter(param_ns + ".weight_timeout_ms", timeout_ms);
  node->get_parameter(param_ns + ".features_topic", features_topic);
  node->get_parameter(param_ns + ".fixed_weights", fallback);
  timeout_ = std::chrono::milliseconds(std::max(50, timeout_ms));
  if (!fallback.empty()) {
    const int n = std::min<int>(num_critics_, static_cast<int>(fallback.size()));
    for (int i = 0; i < n; ++i) {
      fallback_weights_[i] = static_cast<float>(fallback[i]);
    }
  }

  if (!norm_path.empty()) {
    loadNorm(norm_path);
  }

  // ── ONNX session ─────────────────────────────────────────────────────────
#ifdef VF_HAS_ONNXRUNTIME
  if (onnx_path_.empty()) {
    RCLCPP_WARN(
      logger_,
      "OnnxWeightProvider: '%s.onnx_path' is empty; using fallback weights.",
      param_ns.c_str());
  } else {
    try {
      Ort::SessionOptions opts;
      opts.SetIntraOpNumThreads(1);
      opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
      impl_->session = std::make_unique<Ort::Session>(
        impl_->env, onnx_path_.c_str(), opts);
      Ort::AllocatorWithDefaultOptions alloc;
      impl_->in_name = std::string(impl_->session->GetInputNameAllocated(0, alloc).get());
      impl_->out_name = std::string(impl_->session->GetOutputNameAllocated(0, alloc).get());
      impl_->in_shape = {1, static_cast<int64_t>(expected_in_dim_)};
      impl_->ready = true;
      RCLCPP_INFO(
        logger_, "OnnxWeightProvider: loaded '%s' (in='%s', out='%s', D=%d)",
        onnx_path_.c_str(), impl_->in_name.c_str(),
        impl_->out_name.c_str(), expected_in_dim_);
    } catch (const std::exception & e) {
      RCLCPP_WARN(
        logger_, "OnnxWeightProvider: ONNX load failed: %s. Using fallback.",
        e.what());
      impl_->ready = false;
    }
  }
#else
  if (!warned_no_runtime_once_) {
    RCLCPP_WARN(
      logger_,
      "OnnxWeightProvider: built without onnxruntime support. "
      "Falling back to fixed weights. Install onnxruntime or set "
      "weight_provider:='topic' to use the Python sidecar.");
    warned_no_runtime_once_ = true;
  }
  impl_->ready = false;
#endif

  // ── Feature subscriber ──────────────────────────────────────────────────
  rclcpp::QoS qos(rclcpp::KeepLast(10));
  qos.best_effort();
  sub_ = node->create_subscription<std_msgs::msg::Float32MultiArray>(
    features_topic, qos,
    [this](std_msgs::msg::Float32MultiArray::SharedPtr m) { onFeaturesMsg(m); });

  RCLCPP_INFO(
    logger_,
    "OnnxWeightProvider: subscribed to '%s' (timeout=%d ms, K=%d, D=%d, ready=%d)",
    features_topic.c_str(), timeout_ms, num_critics_, expected_in_dim_,
    impl_->ready ? 1 : 0);
}

void OnnxWeightProvider::onFeaturesMsg(
  const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  std::lock_guard<std::mutex> g(mu_);
  latest_features_.assign(msg->data.begin(), msg->data.end());
  if (auto node = node_weak_.lock()) {
    latest_stamp_ = node->now();
  }
  warned_stale_once_ = false;
}

void OnnxWeightProvider::loadNorm(const std::string & path)
{
  // Tiny hand-rolled JSON parser for the two known fields. Avoids pulling
  // a JSON dependency just for this two-array file.
  std::ifstream f(path);
  if (!f.good()) {
    RCLCPP_WARN(logger_, "OnnxWeightProvider: cannot open norm '%s'", path.c_str());
    return;
  }
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string text = ss.str();

  auto extract = [&](const std::string & key) -> std::vector<float> {
    std::vector<float> out;
    const std::string needle = "\"" + key + "\"";
    auto p = text.find(needle);
    if (p == std::string::npos) return out;
    p = text.find('[', p);
    if (p == std::string::npos) return out;
    auto q = text.find(']', p);
    if (q == std::string::npos) return out;
    std::stringstream ns(text.substr(p + 1, q - p - 1));
    std::string tok;
    while (std::getline(ns, tok, ',')) {
      try {
        out.push_back(std::stof(tok));
      } catch (...) {}
    }
    return out;
  };
  norm_mean_ = extract("mean");
  norm_std_ = extract("std");
  if (norm_mean_.empty() || norm_std_.empty() ||
      norm_mean_.size() != norm_std_.size())
  {
    RCLCPP_WARN(
      logger_, "OnnxWeightProvider: norm '%s' shape mismatch (mean=%zu std=%zu); ignoring.",
      path.c_str(), norm_mean_.size(), norm_std_.size());
    norm_mean_.clear();
    norm_std_.clear();
  } else {
    RCLCPP_INFO(
      logger_, "OnnxWeightProvider: loaded norm '%s' (D=%zu)",
      path.c_str(), norm_mean_.size());
  }
}

void OnnxWeightProvider::normalize(std::vector<float> & x) const
{
  if (norm_mean_.empty() || norm_mean_.size() != x.size()) return;
  for (size_t i = 0; i < x.size(); ++i) {
    const float s = norm_std_[i] > 1e-6f ? norm_std_[i] : 1.0f;
    x[i] = (x[i] - norm_mean_[i]) / s;
    if (!std::isfinite(x[i])) x[i] = 0.0f;
  }
}

bool OnnxWeightProvider::isModelLoaded() const
{
  return impl_ && impl_->ready;
}

std::vector<float> OnnxWeightProvider::runForward(const std::vector<float> & x)
{
  (void)x;
#ifdef VF_HAS_ONNXRUNTIME
  if (!impl_ || !impl_->ready || !impl_->session) return {};
  try {
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto in_tensor = Ort::Value::CreateTensor<float>(
      mem, const_cast<float *>(x.data()), x.size(),
      impl_->in_shape.data(), impl_->in_shape.size());
    const char * in_names[]  = {impl_->in_name.c_str()};
    const char * out_names[] = {impl_->out_name.c_str()};
    auto outputs = impl_->session->Run(
      Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);
    if (outputs.empty()) return {};
    const float * out_data = outputs.front().GetTensorData<float>();
    auto info = outputs.front().GetTensorTypeAndShapeInfo();
    const size_t total = info.GetElementCount();
    std::vector<float> result(out_data, out_data + total);
    return result;
  } catch (const std::exception & e) {
    RCLCPP_WARN_THROTTLE(
      logger_, *rclcpp::Clock::make_shared(), 5000,
      "OnnxWeightProvider: forward exception: %s", e.what());
    return {};
  }
#else
  return {};
#endif
}

int OnnxWeightProvider::numCritics() const { return num_critics_; }

std::vector<float> OnnxWeightProvider::getWeights(
  const Eigen::Ref<const Eigen::VectorXf> & features)
{
  // Prefer the latest /vf/features cache, but accept an inline `features`
  // vector as well — the controller passes its own feature buffer through
  // the IWeightProvider API. We pick the larger one (the topic cache is the
  // one that matches expected_in_dim from training).
  std::vector<float> x;
  {
    std::lock_guard<std::mutex> g(mu_);
    auto node = node_weak_.lock();
    bool stale = false;
    if (latest_features_.empty()) {
      stale = true;
    } else if (node) {
      const auto age = node->now() - latest_stamp_;
      if (age > rclcpp::Duration(timeout_)) stale = true;
    }
    if (!stale) {
      x = latest_features_;
    } else if (features.size() == expected_in_dim_) {
      x.assign(features.data(), features.data() + features.size());
    } else {
      if (!warned_stale_once_) {
        RCLCPP_WARN(
          logger_,
          "OnnxWeightProvider: no fresh features within %ld ms; using fallback.",
          static_cast<long>(timeout_.count()));
        warned_stale_once_ = true;
      }
      return fallback_weights_;
    }
  }

  if (static_cast<int>(x.size()) != expected_in_dim_) {
    if (!warned_stale_once_) {
      RCLCPP_WARN(
        logger_,
        "OnnxWeightProvider: feature dim %zu != expected %d; using fallback.",
        x.size(), expected_in_dim_);
      warned_stale_once_ = true;
    }
    return fallback_weights_;
  }

  normalize(x);
  auto raw = runForward(x);
  if (static_cast<int>(raw.size()) < num_critics_) {
    return fallback_weights_;
  }
  std::vector<float> out(num_critics_, 1.0f);
  for (int i = 0; i < num_critics_; ++i) {
    // Clamp pathological outputs into a sane band — never zero (would
    // silence a critic, see the design notes cost-magnitude rule) nor huge
    // (would dwarf upstream weights).
    float v = raw[i];
    if (!std::isfinite(v)) v = 1.0f;
    if (v < 0.05f) v = 0.05f;
    if (v > 10.0f) v = 10.0f;
    out[i] = v;
  }
  return out;
}

}  // namespace vf_robot_controller::meta_critic
