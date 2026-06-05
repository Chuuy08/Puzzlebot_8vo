from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'puzzlebot_gui'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/proto',  glob('proto/*.proto')),
        ('share/' + package_name + '/web',    glob('web/*') + glob('proto/*.proto')),
        ('share/' + package_name,             ['grpcwebproxy']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jesus',
    maintainer_email='a01369587@tec.mx',
    description='Dashboard web de monitoreo para el PuzzleBot',
    license='MIT',
    entry_points={
        'console_scripts': [
            'gui_bridge = puzzlebot_gui.grpc_server:main',
        ],
    },
)
