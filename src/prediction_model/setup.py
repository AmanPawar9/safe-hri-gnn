from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'prediction_model'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files if any
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Include config files if any
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Include model files
        (os.path.join('share', package_name, 'models'), glob('models/*.pth')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='HRI Team',
    maintainer_email='user@example.com',
    description='ST-GNN model for human motion prediction',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'predict_node = prediction_model.predict_node:main',
            'train_model = prediction_model.train:main',
        ],
    },
)
