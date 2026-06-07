"""
mission_launch.py — Lanza el orquestador de misión autónoma.

Requiere correr junto con (en otras terminales):
  ros2 launch puzzlebot_localisation mcl_real_launch.py map:=/path/to/map.yaml
  ros2 launch puzzlebot_navigation navigation_real_launch.py
  (+ los nodos de visión y fpga_controller_node en la Jetson)
"""

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    waypoints_yaml = os.path.join(
        get_package_share_directory('puzzlebot_mission'), 'config', 'waypoints.yaml')

    mission_manager_node = Node(
        package='puzzlebot_mission',
        executable='mission_manager_node',
        name='mission_manager_node',
        output='screen',
        parameters=[{
            'waypoints_yaml_path':       waypoints_yaml,
            'sweep_range_deg':           60.0,
            'sweep_angular_speed':       0.3,
            'sweep_align_threshold_deg': 3.0,
            'sweep_settle_time_s':       1.0,
            'sweep_samples_per_stop':    5,
            'nav_timeout_s':             90.0,
            'fpga_settle_time_s':        5.0,
            'delivery_inflation_radius': 0.02,
            'delivery_robot_radius':     0.10,
            'pallet_inflation_radius':   0.10,
            'costmap_settle_time_s':     1.0,
        }]
    )

    return LaunchDescription([mission_manager_node])
