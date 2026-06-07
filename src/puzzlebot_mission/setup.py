from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'puzzlebot_mission'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jesus',
    maintainer_email='gonzalez.garcia.albertojesus@gmail.com',
    description='Orquestador de misión autónoma del PuzzleBot',
    license='TODO',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'mission_manager_node = puzzlebot_mission.mission_manager_node:main',
            'waypoint_recorder = puzzlebot_mission.waypoint_recorder:main',
            'vision_faker = puzzlebot_mission.vision_faker:main',
        ],
    },
)
