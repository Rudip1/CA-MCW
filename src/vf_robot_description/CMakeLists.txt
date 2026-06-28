cmake_minimum_required(VERSION 3.8)
project(vf_robot_description)

# ── compiler settings ──────────────────────────────────────────────────────────
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── find dependencies ──────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)   # needed for ament_python_install_package
find_package(urdf REQUIRED)
find_package(xacro REQUIRED)

# ── Python module install (the vf_robot_description/ package directory) ────────
# Installs vf_robot_description/__init__.py and any other .py modules
ament_python_install_package(${PROJECT_NAME})

# ── install Python scripts (entry-point scripts, if any) ──────────────────────
# install(PROGRAMS
#   scripts/my_python_node.py
#   DESTINATION lib/${PROJECT_NAME}
# )

# ── install data directories ───────────────────────────────────────────────────
install(
  DIRECTORY
    urdf
    meshes
    launch
    rviz
  DESTINATION share/${PROJECT_NAME}
)

# ── install package marker (required for Python ament packages) ────────────────
install(
  FILES resource/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/packages
)

# ── testing ────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
