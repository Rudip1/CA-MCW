// RobotStateChannel — 9 dims from odom. Phase 5.
//   [0..2] vx, vy, wz
//   [3..4] sin(theta), cos(theta)
//   [5]    |v|              (linear speed magnitude)
//   [6..8] ax, ay, alpha    (angular accel)

#include "vf_robot_controller/perception/features/channels/channel_robot_state.hpp"

#include <cmath>

namespace vf_robot_controller::perception {

RobotStateChannel::RobotStateChannel() = default;

void RobotStateChannel::compute(
  const PerceptionState & state, Eigen::Ref<Eigen::VectorXf> out) const
{
  out.setZero();
  out(0) = state.velocity.x();
  out(1) = state.velocity.y();
  out(2) = state.velocity.z();
  out(3) = std::sin(static_cast<float>(state.robot_pose.theta));
  out(4) = std::cos(static_cast<float>(state.robot_pose.theta));
  out(5) = std::sqrt(out(0) * out(0) + out(1) * out(1));
  out(6) = state.acceleration.x();
  out(7) = state.acceleration.y();
  out(8) = state.acceleration.z();
}

}  // namespace vf_robot_controller::perception
