// Channel: slam_persistent, 40 dims. Phase 6.
//
// Layout (offsets relative to channel start):
//    0..11   topology block   (12 dims)
//   12..27   persistent obstacle rosette (16 dims = 8 angles × 2 radii)
//   28..37   3D structure block (10 dims)
//   38..39   backend capability flags (topology_ok, structure_3d_ok)
//
// Each block zero-fills silently when the active backend lacks the
// corresponding capability, per docs/interfaces.md. The capability flags
// (slots 38, 39) tell the downstream MLP whether the data is real.

#ifndef VF_ROBOT_CONTROLLER__PERCEPTION__FEATURES__CHANNELS__CHANNEL_SLAM_PERSISTENT_HPP_
#define VF_ROBOT_CONTROLLER__PERCEPTION__FEATURES__CHANNELS__CHANNEL_SLAM_PERSISTENT_HPP_

#include <chrono>
#include <mutex>
#include <vector>

#include "vf_robot_controller/perception/features/i_feature_channel.hpp"

namespace vf_robot_controller::perception {

class SlamPersistentChannel : public IFeatureChannel {
public:
  SlamPersistentChannel();
  std::string name() const override { return "slam_persistent"; }
  int dim() const override { return 40; }
  void compute(const PerceptionState & state, Eigen::Ref<Eigen::VectorXf> out) const override;

  // Test-only: bypass the staleness cache (force the next compute() to
  // re-query the backend even if the previous result was fresh).
  void invalidateCacheForTest();

private:
  // Cache the previous compute() output keyed by robot pose. The
  // feature_extractor_node ticks at 20 Hz; backend queries can take
  // several ms, so we serve the cached result if both:
  //   (a) the call is within `cache_ttl_` of the previous,
  //   (b) the robot pose hasn't moved more than `cache_pose_eps_` metres.
  // This is the "never block control loop on perception" pattern. When
  // the cache is hot and pose-stable we skip the backend call entirely.
  mutable std::mutex mu_;
  mutable std::chrono::steady_clock::time_point last_query_{};
  mutable double last_x_{0.0};
  mutable double last_y_{0.0};
  mutable double last_theta_{0.0};
  mutable bool   cache_primed_{false};
  mutable Eigen::VectorXf cache_;
  std::chrono::milliseconds cache_ttl_{500};
  double cache_pose_eps_{0.10};  // metres
};

}  // namespace vf_robot_controller::perception

#endif
