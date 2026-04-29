from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'puzzlebot_description'

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
        (os.path.join('share', package_name, 'models/urdf'), glob(os.path.join('models/urdf', '*.urdf'))),
        (os.path.join('share', package_name, 'models/meshes'), glob(os.path.join('models/meshes', '*.stl'))),
        (os.path.join('share', package_name, 'models/worlds'), glob(os.path.join('models/worlds', '*.sdf'))),
        (os.path.join('share', package_name, 'models/plugins'), glob(os.path.join('models/plugins', '*.so'))),
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
            'joint_state_publisher = puzzlebot_description.joint_state_publisher:main'
        ],
    },
)
