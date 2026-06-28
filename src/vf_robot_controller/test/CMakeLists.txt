# test/CMakeLists.txt — gtest-based unit and integration tests.
# Phase 0: most files are stubs that contain a single passing test so the
# test target builds. Implementations land in the phase named in each file.

find_package(ament_cmake_gtest REQUIRED)

# Helper: register a gtest target with common deps.
function(vf_add_gtest target source)
  ament_add_gtest(${target} ${source})
  if(TARGET ${target})
    target_include_directories(${target} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )
    ament_target_dependencies(${target}
      rclcpp
      rclcpp_lifecycle
      nav2_core
      nav2_costmap_2d
      nav2_util
      geometry_msgs
      nav_msgs
      sensor_msgs
      std_msgs
    )
    target_link_libraries(${target}
      vf_perception_lib
      vf_meta_critic_lib
      vf_logging_lib
      Eigen3::Eigen
    )
  endif()
endfunction()

# ── Unit tests ──────────────────────────────────────────────────────────────
# Perception
vf_add_gtest(test_geometry_utils       unit/perception/test_geometry_utils.cpp)
vf_add_gtest(test_ring_buffer          unit/perception/test_ring_buffer.cpp)
vf_add_gtest(test_voxel_filter         unit/perception/pointcloud/test_voxel_filter.cpp)
vf_add_gtest(test_static_map_backend   unit/perception/map_backend/test_static_map_backend.cpp)
vf_add_gtest(test_rtabmap_backend      unit/perception/map_backend/test_rtabmap_backend.cpp)
vf_add_gtest(test_gcf_2d               unit/perception/gcf/test_gcf_2d.cpp)
vf_add_gtest(test_gcf_3d               unit/perception/gcf/test_gcf_3d.cpp)
vf_add_gtest(test_hysteresis           unit/perception/test_hysteresis.cpp)
vf_add_gtest(test_feature_extractor    unit/perception/features/test_feature_extractor.cpp)
vf_add_gtest(test_channel_robot_state  unit/perception/features/test_channel_robot_state.cpp)
vf_add_gtest(test_channel_path_geometry unit/perception/features/test_channel_path_geometry.cpp)
vf_add_gtest(test_channel_gcf_rosette  unit/perception/features/test_channel_gcf_rosette.cpp)
vf_add_gtest(test_channel_context      unit/perception/features/test_channel_context.cpp)
vf_add_gtest(test_channel_critic_history unit/perception/features/test_channel_critic_history.cpp)
vf_add_gtest(test_channel_obstacle_dynamics unit/perception/features/test_channel_obstacle_dynamics.cpp)
vf_add_gtest(test_channel_reynolds     unit/perception/features/test_channel_reynolds.cpp)
vf_add_gtest(test_channel_slam_persistent unit/perception/features/test_channel_slam_persistent.cpp)

# Meta-critic
vf_add_gtest(test_fixed_weight_provider unit/meta_critic/test_fixed_weight_provider.cpp)
vf_add_gtest(test_topic_weight_provider unit/meta_critic/test_topic_weight_provider.cpp)
vf_add_gtest(test_onnx_weight_provider  unit/meta_critic/test_onnx_weight_provider.cpp)
vf_add_gtest(test_imitation_velocity_provider unit/meta_critic/test_imitation_velocity_provider.cpp)

# Critics — Phase 3 tests must link vf_critics_lib (where the custom critics
# are defined) and nav2_mppi_controller (for mppi::critics::CriticFunction).
function(vf_add_critic_gtest target source)
  ament_add_gtest(${target} ${source})
  if(TARGET ${target})
    target_include_directories(${target} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../include)
    ament_target_dependencies(${target}
      rclcpp rclcpp_lifecycle nav2_core nav2_costmap_2d nav2_util
      nav2_mppi_controller geometry_msgs nav_msgs sensor_msgs std_msgs
      vf_robot_messages)
    target_link_libraries(${target}
      vf_critics_lib vf_perception_lib vf_meta_critic_lib Eigen3::Eigen)
  endif()
endfunction()

vf_add_critic_gtest(test_corridor_critic         unit/critics/test_corridor_critic.cpp)
vf_add_critic_gtest(test_volumetric_critic       unit/critics/test_volumetric_critic.cpp)
vf_add_critic_gtest(test_dynamic_obstacle_critic unit/critics/test_dynamic_obstacle_critic.cpp)

# Controller
ament_add_gtest(test_vf_critic_manager unit/controller/test_vf_critic_manager.cpp)
if(TARGET test_vf_critic_manager)
  target_include_directories(test_vf_critic_manager PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include)
  ament_target_dependencies(test_vf_critic_manager
    rclcpp rclcpp_lifecycle nav2_core nav2_costmap_2d nav2_util)
  target_link_libraries(test_vf_critic_manager
    vf_controller_lib vf_meta_critic_lib Eigen3::Eigen)
endif()

# Logging
vf_add_gtest(test_episode_logger unit/logging/test_episode_logger.cpp)

# ── Phase 7: Python data-collection writer (pytest) ────────────────────────
find_package(ament_cmake_pytest QUIET)
if(ament_cmake_pytest_FOUND)
  ament_add_pytest_test(
    test_episode_writer
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/data_collection/test_episode_writer.py
    TIMEOUT 60
  )

  ament_add_pytest_test(
    test_session
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/data_collection/test_session.py
    TIMEOUT 60
  )

  ament_add_pytest_test(
    test_collector_debounce
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/data_collection/test_collector_debounce.py
    TIMEOUT 60
  )

  # Phase 8: log_reader round-trip + (torch-gated) network forward smoke tests.
  ament_add_pytest_test(
    test_log_reader_and_models
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/training/test_log_reader_and_models.py
    TIMEOUT 60
  )

  # Phase 9.1: oracle weight-recovery QP. cvxpy-gated via pytest.importorskip,
  # so this stays green on a cvxpy-less CI image.
  ament_add_pytest_test(
    test_weight_recovery_qp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/training/test_weight_recovery_qp.py
    TIMEOUT 60
  )

  # Phase 9.2: world replay over recorded HDF5 episodes (h5py + numpy only).
  ament_add_pytest_test(
    test_world_reconstructor
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/training/test_world_reconstructor.py
    TIMEOUT 60
  )

  # Phase 9.3: pure-Python replay oracle. cvxpy-gated; bumped timeout to
  # accommodate the small MPPI rollouts the oracle runs per timestep.
  ament_add_pytest_test(
    test_replay_oracle
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/training/test_replay_oracle.py
    TIMEOUT 120
  )

  # Phase 9.4: scripts/run_oracle.py end-to-end CLI smoke. cvxpy-gated;
  # exercises file/directory handling, idempotency, and --force.
  ament_add_pytest_test(
    test_run_oracle_cli
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/training/test_run_oracle_cli.py
    TIMEOUT 120
  )

  # Phase 9.5: oracle-vs-hindsight target wiring (h5py + numpy only).
  ament_add_pytest_test(
    test_oracle_label_wiring
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/training/test_oracle_label_wiring.py
    TIMEOUT 60
  )
endif()

# ── Integration tests ──────────────────────────────────────────────────────

# test_passthrough_matches_mppi: Phase 1 — needs pluginlib + vf_controller_lib
ament_add_gtest(test_passthrough_matches_mppi
  integration/test_passthrough_matches_mppi.cpp)
if(TARGET test_passthrough_matches_mppi)
  target_include_directories(test_passthrough_matches_mppi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include)
  ament_target_dependencies(test_passthrough_matches_mppi
    rclcpp
    rclcpp_lifecycle
    nav2_core
    nav2_costmap_2d
    nav2_mppi_controller
    pluginlib)
  target_link_libraries(test_passthrough_matches_mppi
    vf_controller_lib)
endif()

vf_add_gtest(test_controller_loads          integration/test_controller_loads.cpp)
vf_add_gtest(test_feature_extractor_pipeline integration/test_feature_extractor_pipeline.cpp)
vf_add_gtest(test_gcf_node                   integration/test_gcf_node.cpp)
