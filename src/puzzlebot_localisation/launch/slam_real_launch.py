import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    loc_pkg  = get_package_share_directory('puzzlebot_localisation')
    desc_pkg = get_package_share_directory('puzzlebot_description')

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    use_slam_tb  = LaunchConfiguration('slam_toolbox', default='false')

    urdf_path  = os.path.join(desc_pkg, 'urdf', 'mcr2_robots', 'puzzlebot_jetson_lidar_ed.xacro')
    robot_desc = ParameterValue(Command(['xacro ', urdf_path]), value_type=str)

    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time, 'robot_description': robot_desc}]
    )

    joint_state_pub = Node(
        package='puzzlebot_description',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        parameters=[{'sampling_time': 0.05}]
    )

    # RPLidar publishes frame_id="laser", URDF names it "laser_frame"
    laser_frame_bridge = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='laser_frame_bridge',
        arguments=['0', '0', '0', '0', '0', '0', 'laser_frame', 'laser'],
    )

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

    icp_node = Node(
        package='puzzlebot_localisation',
        executable='icp_node',
        name='icp_node',
        namespace='icp_node',
        output='screen',
        parameters=[{
            'use_sim_time':            use_sim_time,
            'max_iterations':          30,
            'convergence_threshold':   1e-4,
            'max_correspondence_dist': 0.5,
            'downsample_step':         3,
            'min_points':              20,
            'map_frame':               'map',
            'odom_frame':              'odom',
            'base_frame':              'base_footprint',
            'laser_x_offset':          0.0,
            'laser_y_offset':          0.0,
            'use_odom_init':           True,
            'max_rotation_correction_deg': 5.0,
            'keyframe_dist_m':         0.20,
            'keyframe_angle_deg':      10.0,
            'skip_icp_ang_vel_thresh': 0.15,
        }],
        remappings=[('scan', '/scan')]
    )

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

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('slam_toolbox', default_value='false',
                              description='Launch slam_toolbox alongside ICP'),

        robot_state_pub,
        joint_state_pub,
        laser_frame_bridge,
        localisation_node,
        icp_node,
        icp_map_node,
        slam_toolbox_node,
        rviz_node,
    ])
