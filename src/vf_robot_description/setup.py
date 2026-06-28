from setuptools import setup

# This setup.py is required for ament_cmake_python to correctly install
# the Python package (vf_robot_description/__init__.py).
# In a hybrid C++/Python ament_cmake package, setup.py must be present
# alongside CMakeLists.txt.  Do NOT use setup.cfg — ament_cmake_python
# reads setup.py directly.

package_name = 'vf_robot_description'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        # Registers the package with ament index
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        # Installs package.xml so ros2 pkg can find it
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Pravin Oli',
    maintainer_email='olipravin18@gmail.com',
    description='ViroFighter UVC1 robot description — Python component',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        # Add Python node entry points here if needed, e.g.:
        # 'console_scripts': [
        #     'my_node = vf_robot_description.my_node:main',
        # ],
        'console_scripts': [],
    },
)
