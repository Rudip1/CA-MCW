cmake_minimum_required(VERSION 3.8)
project(vf_robot_gazebo)

# ==============================================================================
# HYBRID PACKAGE — C++ nodes + Python scripts + Gazebo simulation assets
#
# src/                    C++ nodes (compiled)
# vf_robot_gazebo/        Python module (importable)
#   scripts/              Python executables → ros2 run vf_robot_gazebo <name>
# include/                C++ headers
# launch/ models/ worlds/ rviz/   data assets (installed to share/)
#
# urdf/ lives in vf_robot_description — NOT here (single source of truth)
# ==============================================================================

# ------------------------------------------------------------------------------
# C++ Standard
# ------------------------------------------------------------------------------
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)   # hybrid C++/Python support

find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)          # sensor_msgs/Range for ultrasound
find_package(geometry_msgs REQUIRED)        # geometry_msgs/Twist for cmd_vel
find_package(nav_msgs REQUIRED)             # nav_msgs/Odometry
find_package(tf2 REQUIRED)

find_package(gazebo REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(gazebo_ros_pkgs REQUIRED)

find_package(vf_robot_messages REQUIRED)    # custom UltraSound.msg
find_package(vf_robot_description REQUIRED) # URDF + meshes — not duplicated here

include_directories(include ${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})

set(COMMON_DEPENDENCIES
  rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs tf2
  vf_robot_messages
)

# ------------------------------------------------------------------------------
# C++ Executables
# ------------------------------------------------------------------------------

# Aggregates ultrasonic Range msgs and republishes as vf_robot_messages/UltraSound
add_executable(ultrasound_cpp src/ultrasound.cpp)
ament_target_dependencies(ultrasound_cpp ${COMMON_DEPENDENCIES})
target_include_directories(ultrasound_cpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# ------------------------------------------------------------------------------
# Python Scripts Macro
# ------------------------------------------------------------------------------
# Installs every .py in a folder as a ros2 run executable (strips .py extension).
# No __init__.py needed inside script folders.
# To add a new folder: install_python_scripts("vf_robot_gazebo/myfolder")
macro(install_python_scripts FOLDER)
  file(GLOB _SCRIPTS "${FOLDER}/*.py")
  foreach(_SCRIPT ${_SCRIPTS})
    get_filename_component(_NAME ${_SCRIPT} NAME_WE)
    install(PROGRAMS ${_SCRIPT}
      DESTINATION lib/${PROJECT_NAME}
      RENAME ${_NAME}
    )
  endforeach()
endmacro()

# Add script folders here — one line per folder
install_python_scripts("vf_robot_gazebo/scripts")
# install_python_scripts("vf_robot_gazebo/tools")   # uncomment when ready
# install_python_scripts("vf_robot_gazebo/utils")   # uncomment when ready

# ------------------------------------------------------------------------------
# Install
# ------------------------------------------------------------------------------

# C++ executable → ros2 run vf_robot_gazebo ultrasound_cpp
install(TARGETS ultrasound_cpp
  DESTINATION lib/${PROJECT_NAME}
)

# C++ headers
install(DIRECTORY include/
  DESTINATION include
)

# Python module → import vf_robot_gazebo
ament_python_install_package(${PROJECT_NAME})

# Data assets — install(DIRECTORY) is recursive, catches all mesh subdirectories
install(DIRECTORY
  launch      # launch files
  models      # Gazebo world models + meshes (recursive)
  rviz        # RViz configs
  worlds      # Gazebo .world files
  DESTINATION share/${PROJECT_NAME}
)

# ------------------------------------------------------------------------------
# Exports
# ------------------------------------------------------------------------------
ament_export_include_directories(include)
ament_export_dependencies(
  rclcpp rclpy
  std_msgs sensor_msgs geometry_msgs nav_msgs tf2
  gazebo_ros gazebo_ros_pkgs
  vf_robot_messages
  vf_robot_description
)

ament_package()
