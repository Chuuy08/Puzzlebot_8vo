from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'puzzlebot_navigation'

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jesus',
    maintainer_email='gonzalez.garcia.albertojesus@gmail.com',
    description='Navegación autónoma del PuzzleBot',
    license='TODO',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'costmap_node       = puzzlebot_navigation.costmap_node:main',
            'rrt_node           = puzzlebot_navigation.rrt_node:main',
            'path_follower_node = puzzlebot_navigation.path_follower_node:main',
            'dwa_node           = puzzlebot_navigation.dwa_node:main',
        ],
    },
)
