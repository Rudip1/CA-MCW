cmake_minimum_required(VERSION 3.8)
project(vf_robot_bringup)

# ──────────────────────────────────────────────────────────────────────────
# Build type — this package is a hybrid ament_cmake + Python.
# ament_cmake     → installs launch/, config/, rviz/, resource/
# ament_cmake_python → installs the vf_robot_bringup Python package
#                      that contains launch_utils/compose_params.py
# ──────────────────────────────────────────────────────────────────────────

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# ── Install the Python package ──
# This installs vf_robot_bringup/ (the Python module directory) into
# install/vf_robot_bringup/lib/python3.X/site-packages/vf_robot_bringup
# so that `from vf_robot_bringup.launch_utils.compose_params import compose`
# works at runtime in bringup_launch.py.
ament_python_install_package(${PROJECT_NAME})

# ── Install launch files ──
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# ── Install config files (Nav2 params, robot profiles, etc.) ──
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# ── Install RViz config files ──
install(DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)

ament_package()
