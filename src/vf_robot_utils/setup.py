from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'vf_robot_utils'


def _nested_launch_data():
    """Install every launch/**/*.py into share/<pkg>/launch/**/ preserving subdirs.

    Required because ros2 launch resolves <pkg> launches from the install
    share/, not the source tree, and nested launch files (e.g. under
    launch/goalposes_collect/, launch/vf_data_training/batch/) won't be
    found unless they're installed at matching subpaths.
    """
    out = []
    for root, _dirs, files in os.walk('launch'):
        py = sorted(os.path.join(root, f) for f in files if f.endswith('.py'))
        if py:
            out.append((os.path.join('share', package_name, root), py))
    return out


setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'scenarios'), glob('scenarios/*.yaml')),
        (os.path.join('share', package_name, 'config'),    glob('config/*.yaml')),
        (os.path.join('share', package_name, 'runs'),      glob('runs/*.csv')),
        (os.path.join('share', package_name, 'runs'),      glob('runs/*.md')),
        *_nested_launch_data(),
    ],
    install_requires=['setuptools', 'scipy'],
    zip_safe=True,
    maintainer='pravin',
    maintainer_email='olipravin18@gmail.com',
    description='Evaluation harness for vf_robot_controller (runner, metrics, analysis).',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            # Tour replay (used by every batch launch under launch/vf_data_{training,evaluation}/batch/).
            'tour_runner   = vf_robot_utils.tools.tour_runner:main',
            # Standalone goal-pose recorder (used by goalposes_collect launches).
            'pose_recorder = vf_robot_utils.tools.pose_recorder:main',
        ],
    },
)
