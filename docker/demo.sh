#!/bin/bash
set -e

echo '═══════════════════════════════════════'
echo '  BUILD'
echo '═══════════════════════════════════════'
colcon build \
  --packages-select \
    puzzlebot_challenge \
    puzzlebot_control \
    puzzlebot_localisation \
    puzzlebot_description \
    puzzlebot_gazebo \
  --event-handlers console_direct+

echo '═══════════════════════════════════════'
echo '  LAUNCH  — multi_puzzlebot'
echo '  Para ver datos en otra terminal:'
echo '    ros2 topic echo /robot1/pose'
echo '    ros2 topic echo /robot1/cmd_vel'
echo '═══════════════════════════════════════'
source install/setup.bash
ros2 launch puzzlebot_control multi_puzzlebot_launch.py gui:=false
