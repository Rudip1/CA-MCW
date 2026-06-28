// IWeightProvider — interface for sources of per-critic weight vectors.
//
// Implementations:
//   FixedWeightProvider  — reads from YAML, returns constant weights
//   TopicWeightProvider  — subscribes to /vf_controller/meta_weights
//   OnnxWeightProvider   — loads .onnx, runs inference in C++
//
// Selected at runtime by the controller via YAML config.

#ifndef VF_ROBOT_CONTROLLER__META_CRITIC__I_WEIGHT_PROVIDER_HPP_
#define VF_ROBOT_CONTROLLER__META_CRITIC__I_WEIGHT_PROVIDER_HPP_

#include <vector>
#include <Eigen/Core>

namespace vf_robot_controller::meta_critic {

class IWeightProvider {
public:
  virtual ~IWeightProvider() = default;

  /// Number of critics this provider produces weights for.
  virtual int numCritics() const = 0;

  /// Return current weight vector. Caller passes optional features for
  /// inference-time providers; FixedWeightProvider ignores it.
  /// Empty vector return means "no override available, use stock weights".
  virtual std::vector<float> getWeights(
    const Eigen::Ref<const Eigen::VectorXf> & features) = 0;

  /// Provider name for logging.
  virtual std::string name() const = 0;
};

}  // namespace vf_robot_controller::meta_critic

#endif
