// Pluginlib registration for the wrapper critics.
//
// The classes live in `namespace mppi::critics` because upstream's
// CriticManager::getFullName() (nav2_mppi_controller/src/critic_manager.cpp)
// hardcodes the prefix "mppi::critics::" when resolving YAML short names.
// We register under that namespace so upstream's plugin loader can find us.

#include <pluginlib/class_list_macros.hpp>

#include "vf_robot_controller/critics/weighted_critic_wrapper.hpp"

PLUGINLIB_EXPORT_CLASS(mppi::critics::WeightedConstraintCritic, mppi::critics::CriticFunction)
PLUGINLIB_EXPORT_CLASS(mppi::critics::WeightedCostCritic, mppi::critics::CriticFunction)
PLUGINLIB_EXPORT_CLASS(mppi::critics::WeightedGoalCritic, mppi::critics::CriticFunction)
PLUGINLIB_EXPORT_CLASS(mppi::critics::WeightedGoalAngleCritic, mppi::critics::CriticFunction)
PLUGINLIB_EXPORT_CLASS(mppi::critics::WeightedPathAlignCritic, mppi::critics::CriticFunction)
PLUGINLIB_EXPORT_CLASS(mppi::critics::WeightedPathAngleCritic, mppi::critics::CriticFunction)
PLUGINLIB_EXPORT_CLASS(mppi::critics::WeightedPathFollowCritic, mppi::critics::CriticFunction)
PLUGINLIB_EXPORT_CLASS(mppi::critics::WeightedPreferForwardCritic, mppi::critics::CriticFunction)
PLUGINLIB_EXPORT_CLASS(mppi::critics::WeightedTwirlingCritic, mppi::critics::CriticFunction)
PLUGINLIB_EXPORT_CLASS(mppi::critics::WeightedVelocityDeadbandCritic, mppi::critics::CriticFunction)
