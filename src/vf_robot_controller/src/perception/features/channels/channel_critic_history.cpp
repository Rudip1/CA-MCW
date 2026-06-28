// CriticHistoryChannel — 30 dims = 10 cycles × 3 stats. Phase 5.
//
// For each of the last 10 cycles in state.critic_history (newest at end),
// emits (mean, max, std) over that cycle's per-critic costs. If fewer than
// 10 samples exist, the trailing slots are zero-filled. If a sample's
// costs vector is empty, that cycle's stats are also zero-filled.
//
// Note: post-M10, vf_fixedwt and vf_inferencewt publish /vf/per_critic_costs
// unconditionally at 20 Hz (MPPI path). vf_imitationwt skips MPPI, so the
// topic is silent and the history stays empty for the entire run. The
// channel handles both populated and empty histories — zero-filled output
// is still a valid input to the meta-critic / imitation MLPs.

#include "vf_robot_controller/perception/features/channels/channel_critic_history.hpp"

#include <algorithm>
#include <cmath>

namespace vf_robot_controller::perception {

namespace {
constexpr int kHistoryLen = 10;
constexpr int kStatsPerCycle = 3;  // mean, max, std
}  // namespace

CriticHistoryChannel::CriticHistoryChannel() = default;

void CriticHistoryChannel::compute(
  const PerceptionState & state, Eigen::Ref<Eigen::VectorXf> out) const
{
  out.setZero();
  const auto & hist = state.critic_history;
  const size_t n = std::min<size_t>(hist.size(), kHistoryLen);
  // Newest sample maps to the *last* slot in the output; oldest to the
  // first occupied slot. This keeps the time axis monotonic regardless of
  // how many samples we have.
  const size_t hist_offset = hist.size() - n;
  const size_t out_offset = kHistoryLen - n;

  for (size_t i = 0; i < n; ++i) {
    const auto & sample = hist[hist_offset + i];
    const auto & costs = sample.costs;
    const int slot = static_cast<int>(out_offset + i) * kStatsPerCycle;
    if (costs.empty()) continue;

    float sum = 0.0f, mx = costs[0];
    for (float c : costs) {
      sum += c;
      mx = std::max(mx, c);
    }
    const float mean = sum / static_cast<float>(costs.size());
    float var = 0.0f;
    for (float c : costs) {
      const float d = c - mean;
      var += d * d;
    }
    var /= static_cast<float>(costs.size());

    out(slot + 0) = mean;
    out(slot + 1) = mx;
    out(slot + 2) = std::sqrt(var);
  }
}

}  // namespace vf_robot_controller::perception
