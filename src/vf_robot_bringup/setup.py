from setuptools import setup

# This setup.py exists only to satisfy ament_cmake_python's
# ament_python_install_package() macro. It is not used for pip/wheel
# distribution — the package is installed by colcon via CMakeLists.txt.
#
# The actual package metadata (name, maintainer, deps) lives in package.xml,
# and the actual install steps live in CMakeLists.txt.

setup(
    name="vf_robot_bringup",
    version="0.1.0",
    packages=[
        "vf_robot_bringup",
        "vf_robot_bringup.launch_utils",
    ],
    install_requires=["setuptools"],
    zip_safe=True,
)
