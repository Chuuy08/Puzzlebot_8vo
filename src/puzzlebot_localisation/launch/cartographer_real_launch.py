import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    loc_pkg  = get_package_share_directory('puzzlebot_localisation')
    desc_pkg = get_package_share_directory('puzzlebot_description')

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    urdf_path  = os.path.join(desc_pkg, 'urdf', 'mcr2_robots', 'puzzlebot_jetson_lidar_ed.xacro')
    robot_desc = ParameterValue(Command(['xacro ', urdf_path]), value_type=str)

    # URDF → TF tree: base_footprint → laser_frame (required by Cartographer)
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time, 'robot_description': robot_desc}]
    )

    # Wheel joint states for visualisation
    joint_state_pub = Node(
        package='puzzlebot_description',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        parameters=[{'sampling_time': 0.05}]
    )

    # RPLidar driver publishes scan with frame_id="laser" but URDF names it "laser_frame".
    # This identity transform bridges the two names.
    laser_frame_bridge = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='laser_frame_bridge',
        arguments=['0', '0', '0', '0', '0', '0', 'laser_frame', 'laser'],
    )

    # Dead-reckoning odometry → odom → base_footprint TF + /odom topic
    # Real robot publishes VelocityEncR/L instead of wr/wl
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
        }],
        remappings=[
            ('wr', 'VelocityEncR'),
            ('wl', 'VelocityEncL'),
        ]
    )

    # Cartographer: scan matching + loop closure → publishes map → odom TF and /map
    cartographer_node = TimerAction(
        period=3.0,
        actions=[Node(
            package='cartographer_ros',
            executable='cartographer_node',
            name='cartographer_node',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=[
                '-configuration_directory', os.path.join(loc_pkg, 'config'),
                '-configuration_basename', 'cartographer.lua',
            ],
            remappings=[
                ('scan', '/scan'),
                ('odom', '/odom'),
            ],
        )]
    )

    # Converts Cartographer submaps to nav_msgs/OccupancyGrid on /map
    cartographer_grid_node = TimerAction(
        period=3.0,
        actions=[Node(
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node',
            name='cartographer_occupancy_grid_node',
            output='screen',
            parameters=[{
                'use_sim_time':       use_sim_time,
                'resolution':         0.05,
                'publish_period_sec': 1.0,
            }],
        )]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),

        robot_state_pub,
        joint_state_pub,
        laser_frame_bridge,
        localisation_node,
        cartographer_node,
        cartographer_grid_node,
        rviz_node,
    ])
