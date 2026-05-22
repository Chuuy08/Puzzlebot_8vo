"""
SLAM Launch File
Package : puzzlebot_localisation
File    : launch/slam_launch.py

Starts:
  1. Gazebo simulation (world + robot)
  2. Wheel-odometry localisation (odom → base_footprint TF)
  3. ICP scan matching node  → /icp_pose, /icp_odom, /icp_map
  4. slam_toolbox             → /map  (for comparison)
  5. RViz with both maps side by side

Usage:
  ros2 launch puzzlebot_localisation slam_launch.py
  ros2 launch puzzlebot_localisation slam_launch.py use_sim_time:=false  # real robot
  ros2 launch puzzlebot_localisation slam_launch.py world:=puzzlebot_arena.world
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Packages ──────────────────────────────────────────────────────
    loc_pkg   = get_package_share_directory('puzzlebot_localisation')
    gazebo_pkg = get_package_share_directory('puzzlebot_gazebo')

    # ── Arguments ─────────────────────────────────────────────────────
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world        = LaunchConfiguration('world', default='obstacle_avoidance_4.world')
    use_slam_tb  = LaunchConfiguration('slam_toolbox', default='true')

    # ── 1. Gazebo simulation ───────────────────────────────────────────
    # Includes: Gazebo world + robot spawn + ros_gz_bridge + joint_vel_bridge
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
            'robot_name':   '',           # no namespace → topics at /scan, /odom
            'prefix':       '',
            'x': '0.0', 'y': '0.0', 'yaw': '0.0',
            'lidar_frame':  'laser_frame',
            'camera_frame': 'camera_link_optical',
            'tof_frame':    'tof_link',
            'use_sim_time': 'true',
        }.items()
    )

    # ── 2. Wheel-odometry localisation ────────────────────────────────
    # Provides TF: odom → base_footprint  and publishes /odom
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

    # ── 3. ICP node ───────────────────────────────────────────────────
    # namespace='icp_node' → topics at /icp_node/icp_pose, /icp_node/icp_odom
    icp_node = Node(
        package='puzzlebot_localisation',
        executable='icp_node',
        name='icp_node',
        namespace='icp_node',
        output='screen',
        parameters=[{
            'use_sim_time':             use_sim_time,
            'max_iterations':           30,
            'convergence_threshold':    1e-4,
            'max_correspondence_dist':  0.5,
            'downsample_step':          3,
            'min_points':               20,
            'map_frame':   'map',
            'base_frame':  'base_footprint',
            'laser_x_offset': 0.0,
            'laser_y_offset': 0.0,
            'use_odom_init': True,
            'max_rotation_correction_deg': 5.0,  # allow slightly larger angle correction
            'keyframe_dist_m':   0.20,   # keyframe every 20 cm (more dense)
            'keyframe_angle_deg': 10.0,  # keyframe every 10° (catches curves better)
        }],
        remappings=[('scan', '/scan')]
    )

    # ── 4. ICP occupancy grid node ────────────────────────────────────
    # Same namespace → subscribes to /icp_node/icp_odom automatically
    icp_map_node = Node(
        package='puzzlebot_localisation',
        executable='icp_map_node',
        name='icp_map_node',
        namespace='icp_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'resolution':   0.05,
            'grid_size':    200,
            'map_frame':    'map',
            'publish_rate': 2.0,
        }],
        remappings=[('scan', '/scan')]
    )

    # ── 5. slam_toolbox (online async) ────────────────────────────────
    # Delayed 5 s so Gazebo + localisation have time to publish TFs first.
    # Provides: /map topic + map → odom TF (graph-optimised, with loop closure)
    slam_toolbox_node = TimerAction(
        period=5.0,
        actions=[Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            condition=IfCondition(use_slam_tb),
            parameters=[
                os.path.join(loc_pkg, 'config', 'slam_toolbox.yaml'),
                {'use_sim_time': use_sim_time}
            ],
        )]
    )

    # ── 6. RViz ───────────────────────────────────────────────────────
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
        DeclareLaunchArgument('slam_toolbox', default_value='true',
                              description='Launch slam_toolbox alongside ICP'),

        gazebo_world,
        spawn_robot,
        localisation_node,
        icp_node,
        icp_map_node,
        slam_toolbox_node,
        rviz_node,
    ])
