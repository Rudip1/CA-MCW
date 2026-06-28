// ContextChannel — 9-dim one-hot of NavigationContext id. Phase 5.
//
// 6 known contexts (OPEN..APPROACHING_GOAL, ids 0..5) + 3 reserved slots
// for future contexts. UNKNOWN (255) zeros all dimensions.

#include "vf_robot_controller/perception/features/channels/channel_context.hpp"

namespace vf_robot_controller::perception {

ContextChannel::ContextChannel() = default;

void ContextChannel::compute(
  const PerceptionState & state, Eigen::Ref<Eigen::VectorXf> out) const
{
  out.setZero();
  if (state.context_id < 9) {
    out(state.context_id) = 1.0f;
  }
}

}  // namespace vf_robot_controller::perception
