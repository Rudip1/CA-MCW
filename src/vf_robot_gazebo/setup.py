from setuptools import setup
from glob import glob

# In a HYBRID ament_cmake + Python package, setup.py has a LIMITED role:
#   - Registers the Python module (vf_robot_gazebo/) so `import vf_robot_gazebo` works
#   - Installs flat data files (launch, rviz, worlds)
#
# What CMakeLists.txt handles instead:
#   - C++ compilation and executables
#   - Python scripts → ros2 run  (via file(GLOB) macro, not console_scripts here)
#   - models/ install (recursive, catches all mesh subdirectories)
#   - urdf/ is in vf_robot_description — not here

package_name = "vf_robot_gazebo"

setup(
    name=package_name,
    version="0.1.0",
    # Top-level Python package only — scripts/ has no __init__.py
    # and is handled by CMakeLists.txt install(PROGRAMS) macro
    packages=[package_name],
    data_files=[
        # ament index marker — required for ros2 pkg list / get_package_share_directory
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        # package.xml
        (f"share/{package_name}", ["package.xml"]),
        # Launch files
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        # RViz configs
        (f"share/{package_name}/rviz", glob("rviz/*.rviz")),
        # Gazebo world files
        (f"share/{package_name}/worlds", glob("worlds/*.world")),
        # models/ → handled by CMakeLists.txt install(DIRECTORY) — recursive
        # urdf/   → lives in vf_robot_description — not here
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Pravin",
    maintainer_email="olipravin18@gmail.com",
    description="Gazebo simulation package for VF Robot — standalone, no SLAM/Nav2.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        # Python scripts are installed via CMakeLists.txt file(GLOB) macro.
        # Any .py file in vf_robot_gazebo/scripts/ becomes a ros2 run executable
        # automatically — no entries needed here.
        "console_scripts": [],
    },
)
