import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'puzzlebot_voice'


def media_data_files():
    """Recursively register every .wav under media/audio so it gets installed
    into share/<package>/media/audio/<word>/... preserving the folder layout."""
    data_files = []
    for root, _dirs, files in os.walk('media/audio'):
        wavs = [os.path.join(root, f) for f in files if f.endswith('.wav')]
        if wavs:
            data_files.append((os.path.join('share', package_name, root), wavs))
    return data_files


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob(os.path.join('models', '*.pkl'))),
    ] + media_data_files(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jesus',
    maintainer_email='gonzalez.garcia.albertojesus@gmail.com',
    description='Voice command recognition for the Puzzlebot (HMM-based, microphone capture and /voice_cmd publisher)',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'voice_recognition = puzzlebot_voice.voice_recognition:main',
            'voice_cmd = puzzlebot_voice.voice_cmd:main',
            'grabar = puzzlebot_voice.grabar:main',
            'hmm_heatmaps = puzzlebot_voice.hmm_heatmaps:main',
            'legacy_voice_cmd = puzzlebot_voice.legacy_voice_cmd.main:main',
        ],
    },
)
