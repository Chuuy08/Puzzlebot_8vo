from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'tf_examples'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
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
            'static_tf = tf_examples.static_tf:main',
            'dyn_tf = tf_examples.dyn_tf:main',
            'tf_listener = tf_examples.tf_listener:main',
        ],
    },
)

