from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'robot_control'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Include config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Include URDF/SRDF files if any
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'srdf'), glob('srdf/*.srdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='HRI Team',
    maintainer_email='user@example.com',
    description='Robot control module with MoveIt 2 integration for collision avoidance',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motion_planner_node = robot_control.motion_planner_node:main',
        ],
    },
)
