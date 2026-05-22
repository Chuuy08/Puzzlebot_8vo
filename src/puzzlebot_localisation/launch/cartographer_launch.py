"""
Cartographer SLAM Launch File
Package : puzzlebot_localisation
File    : launch/cartographer_launch.py

Starts:
  1. Gazebo simulation (world + robot)
  2. Wheel-odometry localisation  → odom → base_footprint TF
  3. Google Cartographer          → /map  +  map → odom TF  (with loop closure)
  4. Cartographer occupancy grid  → /map  as OccupancyGrid
  5. RViz

Usage:
  ros2 launch puzzlebot_localisation cartographer_launch.py
  ros2 launch puzzlebot_localisation cartographer_launch.py use_sim_time:=false
  ros2 launch puzzlebot_localisation cartographer_launch.py world:=puzzlebot_arena.world
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Packages ──────────────────────────────────────────────────────
    loc_pkg    = get_package_share_directory('puzzlebot_localisation')
    gazebo_pkg = get_package_share_directory('puzzlebot_gazebo')

    # ── Arguments ─────────────────────────────────────────────────────
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world        = LaunchConfiguration('world', default='obstacle_avoidance_4.world')

    # ── 1. Gazebo simulation ──────────────────────────────────────────
    gazebo_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_pkg, 'launch', 'gazebo_world_launch.py')),
        launch_arguments={'world': world, 'pause': 'false'}.items()
    )

    spawn_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_pkg, 'launch', 'gazebo_puzzlebot_launch.py')),
        launch_arguments={
            'robot':        'puzzlebot_jetson_lidar_ed',
            'robot_name':   '',
            'prefix':       '',
            'x': '0.0', 'y': '0.0', 'yaw': '0.0',
            'lidar_frame':  'laser_frame',
            'camera_frame': 'camera_link_optical',
            'tof_frame':    'tof_link',
            'use_sim_time': 'true',
        }.items()
    )

    # ── 2. Wheel-odometry localisation ────────────────────────────────
    # Provides TF: odom → base_footprint  and topic /odom
    # Cartographer consumes both.
    localisation_node = Node(
        package='puzzlebot_localisation',
        executable='localisation',
        name='localisation',
        output='screen',
        parameters=[{
            'use_sim_time':  use_sim_time,
            'wheel_radius':  0.05,
            'wheel_base':    0.19,
            'sampling_time': 0.05,
            'k_r': 0.1592,
            'k_l': 0.2128,
            'odom_frame': 'odom',
        }]
    )

    # ── 3. Google Cartographer ─────────────────────────────────────────
    # -configuration_directory  path to the folder that contains the .lua file
    # -configuration_basename   the .lua filename (no path)
    #
    # Input topics (remapped to absolute):
    #   /scan   LaserScan
    #   /odom   Odometry  (wheel encoders, use_odometry=true in .lua)
    # Input TF:
    #   odom → base_footprint  (from localisation_node)
    #
    # Output:
    #   /map            SubmapList (internal Cartographer format)
    #   TF map → odom   (corrected, graph-optimised, with loop closure)
    cartographer_node = TimerAction(
        period=3.0,    # wait for Gazebo + localisation TF to be ready
        actions=[Node(
            package='cartographer_ros',
            executable='cartographer_node',
            name='cartographer_node',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=[
                '-configuration_directory',
                os.path.join(loc_pkg, 'config'),
                '-configuration_basename', 'cartographer.lua',
            ],
            remappings=[
                ('scan', '/scan'),
                ('odom', '/odom'),
            ],
        )]
    )

    # ── 4. Cartographer occupancy grid ────────────────────────────────
    # Converts Cartographer submaps to a standard nav_msgs/OccupancyGrid
    # published on /map — the same topic RViz and nav_stack expect.
    cartographer_grid_node = TimerAction(
        period=3.0,
        actions=[Node(
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node',
            name='cartographer_occupancy_grid_node',
            output='screen',
            parameters=[{
                'use_sim_time':      use_sim_time,
                'resolution':        0.01,  # 1cm grid cells
                'publish_period_sec': 1.0,
            }],
        )]
    )

    # ── 5. RViz ───────────────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='true for Gazebo, false for real robot'),
        DeclareLaunchArgument('world', default_value='obstacle_avoidance_4.world',
                              description='Gazebo world file name'),

        gazebo_world,
        spawn_robot,
        localisation_node,
        cartographer_node,
        cartographer_grid_node,
        rviz_node,
    ])
