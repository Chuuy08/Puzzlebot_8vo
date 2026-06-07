import os

from setuptools import find_packages, setup

package_name = 'puzzlebot_fpga_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('lib/python3.8/site-packages', package_name),
            [package_name + '/libspi_fsm.so']),
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
            'fpga_controller_node = puzzlebot_fpga_controller.fpga_controller_node:main',
        ],
    },
)
