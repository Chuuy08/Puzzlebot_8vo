from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'puzzlebot_localisation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.[yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.lua'))),
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*.rviz'))),
        (os.path.join('share', package_name, 'meshes'), glob(os.path.join('meshes', '*.stl'))),
        (os.path.join('share', package_name, 'urdf'), glob(os.path.join('urdf', '*.urdf'))),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jesus',
    maintainer_email='gonzalez.garcia.albertojesus@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'localisation = puzzlebot_localisation.localisation:main',
            'mcl_node = puzzlebot_localisation.mcl_node:main',
            'icp_node = puzzlebot_localisation.icp_node:main',
            'icp_map_node = puzzlebot_localisation.occupancy_grid_node:main',
        ],
    },
)
