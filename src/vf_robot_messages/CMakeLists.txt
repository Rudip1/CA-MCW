cmake_minimum_required(VERSION 3.5)
project(vf_robot_messages)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Message files
set(msg_files
  msg/ArcMovement.msg
  msg/ChannelValues.msg
  msg/Command.msg
  msg/CorridorPiece.msg
  msg/Cubic.msg
  msg/HallwayRecord.msg
  msg/MoveParameters.msg
  msg/MoveStampedArray.msg
  msg/MoveStamped.msg
  msg/PathPoints.msg
  msg/PlanarCubicSpline.msg
  msg/PossibleManeuvers.msg
  msg/ReportPoint.msg
  msg/RGBLED.msg
  msg/StatusReport.msg
  msg/Trajectory.msg
  msg/UltraSound.msg
  msg/WallPoints.msg
  msg/MppiCriticsStats.msg
  msg/MppiTrajectoryPoint.msg
  msg/MppiTrajectory.msg
)

# Service files
set(srv_files
  srv/CalibrateCamera.srv
  srv/MoveParametersStringType.srv
  srv/PointCloudType.srv
  srv/ServiceStringArrayRequest.srv
  srv/ServiceString.srv
  srv/StartCalibrationIntrinsics.srv
  srv/StringMoveParametersType.srv
  srv/StringPointCloudType.srv
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
