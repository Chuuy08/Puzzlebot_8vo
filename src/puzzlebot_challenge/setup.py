from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'puzzlebot_challenge'

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
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*.rviz'))),
        (os.path.join('share', package_name, 'meshes'), glob(os.path.join('meshes', '*.stl'))),
        (os.path.join('share', package_name, 'urdf'), glob(os.path.join('urdf', '*.urdf'))),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jesus',
    maintainer_email='gonzalez.garcia.albertojesus@gmail.com',
    description='URDFLink Examples',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'puzzlebot = puzzlebot_challenge.puzzlebot:main',
            'puzzlebot_kinematic = puzzlebot_challenge.puzzlebot_kinematic:main',
            'localisation = puzzlebot_challenge.localisation:main',
            'joint_state_publisher = puzzlebot_challenge.joint_state_publisher:main',
            'control = puzzlebot_challenge.control:main',
            'set_poin_generator = puzzlebot_challenge.set_poin_generator:main',
        ],
    },
)
