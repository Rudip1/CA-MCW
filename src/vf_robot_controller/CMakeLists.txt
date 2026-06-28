cmake_minimum_required(VERSION 3.8)
project(vf_robot_controller)

# PCL_QHULL_PATCH_v1: PCL temporarily disabled for Phase 0.
# Reason: PCL's CMake config on Ubuntu 22.04 exports an imported target
# QHULL::QHULL that is undefined unless find_package(Qhull) is called.
# No Phase 0 source uses PCL, so we defer it to Phase 4 (voxel_filter).

# Default to C++17, enable optimization flags
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# ── Find dependencies ──────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav2_mppi_controller REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(angles REQUIRED)
find_package(pluginlib REQUIRED)
find_package(Eigen3 REQUIRED)
# Phase 0: PCL deferred to Phase 4 (voxel_filter implementation).
# Avoids QHULL::QHULL imported-target bug in PCL's CMake config on Ubuntu 22.04.
# When Phase 4 lands, re-enable via pkg-config (sidesteps the broken config):
#   find_package(PkgConfig REQUIRED)
#   pkg_check_modules(PCL_PKGCONFIG REQUIRED pcl_common pcl_io pcl_filters)
# find_package(PCL REQUIRED COMPONENTS common io filters)
# find_package(pcl_ros REQUIRED)  # Deferred to Phase 4 (transitively pulls broken PCL config)
# find_package(pcl_conversions REQUIRED)  # Deferred to Phase 4 (transitively pulls broken PCL config)
find_package(SQLite3 REQUIRED)
find_package(vf_robot_messages REQUIRED)

set(COMMON_DEPENDENCIES
  rclcpp
  rclcpp_lifecycle
  nav2_core
  nav2_costmap_2d
  nav2_util
  nav2_mppi_controller
  nav_msgs
  geometry_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  angles
  pluginlib
  vf_robot_messages
)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  # ${PCL_INCLUDE_DIRS}  # Phase 4
)

# ───────────────────────────────────────────────────────────────────────────
# Library 1: vf_perception_lib
#   GCF, context classifier, feature extractor, map backends, voxel filter.
#   Depends on nothing in this package. Pure perception.
# ───────────────────────────────────────────────────────────────────────────
add_library(vf_perception_lib SHARED
  src/perception/common/geometry_utils.cpp
  src/perception/pointcloud/voxel_filter.cpp
  src/perception/gcf/gcf_2d.cpp
  src/perception/gcf/gcf_3d.cpp
  src/perception/gcf/gcf_composite.cpp
  src/perception/gcf/corridor_analyzer.cpp
  src/perception/gcf/clutter_detector.cpp
  src/perception/context/hysteresis_classifier.cpp
  src/perception/map_backend/static_map_backend.cpp
  src/perception/map_backend/rtabmap_backend.cpp
  src/perception/map_backend/cuvslam_backend.cpp
  src/perception/features/feature_extractor.cpp
  src/perception/features/channels/channel_robot_state.cpp
  src/perception/features/channels/channel_path_geometry.cpp
  src/perception/features/channels/channel_gcf_rosette.cpp
  src/perception/features/channels/channel_obstacle_dynamics.cpp
  src/perception/features/channels/channel_reynolds.cpp
  src/perception/features/channels/channel_critic_history.cpp
  src/perception/features/channels/channel_context.cpp
  src/perception/features/channels/channel_slam_persistent.cpp
)
ament_target_dependencies(vf_perception_lib ${COMMON_DEPENDENCIES})
target_link_libraries(vf_perception_lib
  Eigen3::Eigen
  # ${PCL_LIBRARIES}  # Phase 4
  SQLite::SQLite3
)
target_include_directories(vf_perception_lib PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)

# ── onnxruntime (optional) ────────────────────────────────────────────────
# Enables OnnxWeightProvider's ONNX inference path.  Without it the provider
# compiles but silently falls back to fixed weights at runtime.
# Install: download onnxruntime-linux-x64-<ver>.tgz from GitHub releases,
#   copy include/* to /usr/local/include/ and add a libonnxruntime.so symlink
#   to /usr/local/lib/, then rebuild this package.
find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  PATHS /usr/local/include /usr/include
)
find_library(ONNXRUNTIME_LIB
  NAMES onnxruntime
  PATHS /usr/local/lib /usr/lib
)
if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
  message(STATUS "onnxruntime found: ${ONNXRUNTIME_LIB} — OnnxWeightProvider ONNX path enabled")
  set(VF_HAS_ONNXRUNTIME TRUE)
else()
  message(WARNING
    "onnxruntime NOT found — OnnxWeightProvider will fall back to fixed weights. "
    "See CMakeLists.txt comment above for install steps.")
  set(VF_HAS_ONNXRUNTIME FALSE)
endif()

# ───────────────────────────────────────────────────────────────────────────
# Library 2: vf_meta_critic_lib
#   Weight providers (Fixed, Topic, Onnx). Tiny.
# ───────────────────────────────────────────────────────────────────────────
add_library(vf_meta_critic_lib SHARED
  src/meta_critic/fixed_weight_provider.cpp
  src/meta_critic/topic_weight_provider.cpp
  src/meta_critic/onnx_weight_provider.cpp
  src/meta_critic/imitation_velocity_provider.cpp
)
ament_target_dependencies(vf_meta_critic_lib ${COMMON_DEPENDENCIES})
target_include_directories(vf_meta_critic_lib PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)
if(VF_HAS_ONNXRUNTIME)
  target_compile_definitions(vf_meta_critic_lib PRIVATE VF_HAS_ONNXRUNTIME)
  target_include_directories(vf_meta_critic_lib PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(vf_meta_critic_lib ${ONNXRUNTIME_LIB})
endif()

# ───────────────────────────────────────────────────────────────────────────
# Library 3: vf_critics_lib
#   Custom critics + Phase 2 weight-injection wrappers around upstream
#   critics. All inherit from mppi::critics::CriticFunction (directly or
#   transitively) and load via upstream's CriticManager.
# ───────────────────────────────────────────────────────────────────────────
add_library(vf_critics_lib SHARED
  src/critics/corridor_critic.cpp
  src/critics/volumetric_critic.cpp
  src/critics/dynamic_obstacle_critic.cpp
  src/critics/weighted_critic_wrapper.cpp
)
ament_target_dependencies(vf_critics_lib ${COMMON_DEPENDENCIES})
target_link_libraries(vf_critics_lib
  vf_perception_lib
  Eigen3::Eigen
)
target_include_directories(vf_critics_lib PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)

# ───────────────────────────────────────────────────────────────────────────
# Library 4: vf_logging_lib
#   HDF5 episode logger.
# ───────────────────────────────────────────────────────────────────────────
add_library(vf_logging_lib SHARED
  src/logging/episode_logger.cpp
  src/logging/synchronizer.cpp
)
ament_target_dependencies(vf_logging_lib ${COMMON_DEPENDENCIES})
target_include_directories(vf_logging_lib PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)

# ───────────────────────────────────────────────────────────────────────────
# Library 5: vf_controller_lib
#   The thin shim on nav2_mppi_controller. This is the plugin entry point.
# ───────────────────────────────────────────────────────────────────────────
add_library(vf_controller_lib SHARED
  src/controller/vf_controller.cpp
  src/controller/vf_critic_manager.cpp
)
ament_target_dependencies(vf_controller_lib ${COMMON_DEPENDENCIES})
target_link_libraries(vf_controller_lib
  vf_perception_lib
  vf_meta_critic_lib
  vf_logging_lib
  Eigen3::Eigen
)
target_include_directories(vf_controller_lib PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)

# ───────────────────────────────────────────────────────────────────────────
# Node executables
# ───────────────────────────────────────────────────────────────────────────
add_executable(gcf_node              nodes/gcf_node.cpp)
add_executable(feature_extractor_node nodes/feature_extractor_node.cpp)
add_executable(context_node          nodes/context_node.cpp)
# Phase 7: data_collector_node is a Python sidecar (vf_controller/data_collection/
# data_collector_node.py). The C++ stub at nodes/data_collector_node.cpp is kept
# in-tree as a placeholder but is no longer built — Phase 7 chose the Python
# implementation for h5py / numpy convenience over the HDF5 C++ API.
add_executable(map_backend_node      nodes/map_backend_node.cpp)

foreach(node_target gcf_node feature_extractor_node context_node map_backend_node)
  ament_target_dependencies(${node_target} ${COMMON_DEPENDENCIES})
  target_link_libraries(${node_target} vf_perception_lib vf_logging_lib)
endforeach()

# ───────────────────────────────────────────────────────────────────────────
# Install
# ───────────────────────────────────────────────────────────────────────────
install(
  TARGETS
    vf_perception_lib
    vf_meta_critic_lib
    vf_critics_lib
    vf_logging_lib
    vf_controller_lib
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  TARGETS
    gcf_node
    feature_extractor_node
    context_node
    map_backend_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})
install(DIRECTORY launch   DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config   DESTINATION share/${PROJECT_NAME})
install(DIRECTORY plugins  DESTINATION share/${PROJECT_NAME})
install(DIRECTORY models   DESTINATION share/${PROJECT_NAME}
        PATTERN "_archive_*" EXCLUDE)

install(FILES plugins/controller_plugins.xml plugins/critic_plugins.xml
        DESTINATION share/${PROJECT_NAME})

# Python sub-package (vf_controller/)
ament_python_install_package(vf_controller)

install(PROGRAMS
  vf_controller/inference/metacritic_inference_node.py
  vf_controller/inference/imitation_inference_node.py
  vf_controller/data_collection/data_collector_node.py
  vf_controller/data_collection/session_status.py
  scripts/train.py
  scripts/export_model.py
  scripts/compute_normalization.py
  scripts/run_oracle.py
  scripts/vf_session_status
  DESTINATION lib/${PROJECT_NAME}
)

# Pluginlib export
pluginlib_export_plugin_description_file(nav2_core plugins/controller_plugins.xml)
pluginlib_export_plugin_description_file(nav2_mppi_controller plugins/critic_plugins.xml)

# ── Tests ──────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
  add_subdirectory(test)
endif()

ament_export_libraries(
  vf_perception_lib
  vf_meta_critic_lib
  vf_critics_lib
  vf_logging_lib
  vf_controller_lib
)
ament_export_include_directories(include)
ament_export_dependencies(${COMMON_DEPENDENCIES} Eigen3)
ament_export_targets(export_${PROJECT_NAME})

ament_package()
