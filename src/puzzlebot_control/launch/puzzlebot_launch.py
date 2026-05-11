import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():

    robot_params = {
        'wheel_radius':  0.05,
        'wheel_base':    0.19,
        'sampling_time': 0.05,
    }

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    urdf_path = os.path.join(
        get_package_share_directory('puzzlebot_description'),
        'urdf', 'mcr2_robots',
        'puzzlebot_jetson_lidar_ed.xacro')

    robot_desc = ParameterValue(Command(['xacro ', urdf_path]), value_type=str)

    kinematic_model_node = Node(
        package='puzzlebot_challenge',
        executable='puzzlebot_kinematic',
        name='puzzlebot_kinematic',
        output='screen',
        parameters=[robot_params]
    )

    localisation_node = Node(
        package='puzzlebot_localisation',
        executable='localisation',
        name='localisation',
        output='screen',
        parameters=[robot_params]
    )

    joint_state_node = Node(
        package='puzzlebot_description',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        parameters=[{'sampling_time': robot_params['sampling_time']}]
    )

    robot_state_pub_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time, 'robot_description': robot_desc}]
    )

    control_node = Node(
        package='puzzlebot_control',
        executable='control',
        name='control',
        output='screen',
        parameters=[{
            'Kp_d': 0.25,
            'Ki_d': 0.0,
            'Kd_d': 0.0,
            'Kp_theta': 1.8,
            'Ki_theta': 0.0,
            'Kd_theta': 0.0,
            'threshold': 0.1,
            'sampling_time': 0.05,
        }]
    )

    setpoint_generator_node = Node(
        package='puzzlebot_control',
        executable='set_poin_generator',
        name='setpoint_generator',
        output='screen',
        parameters=[{
            'trajectory': 'square',
            'side_length': 1.0,
            'loop': False,
        }]
    )

    static_transform_node_map_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--yaw', '0', '--pitch', '0', '--roll', '0',
                   '--frame-id', 'map', '--child-frame-id', 'odom']
    )

    static_transform_node_world_map = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--yaw', '0', '--pitch', '0', '--roll', '0',
                   '--frame-id', 'world', '--child-frame-id', 'map']
    )

    rqt_tf_tree_node = Node(
        name='rqt_tf_tree',
        package='rqt_tf_tree',
        executable='rqt_tf_tree'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),

        static_transform_node_map_odom,
        static_transform_node_world_map,
        robot_state_pub_node,
        kinematic_model_node,
        localisation_node,
        joint_state_node,
        control_node,
        setpoint_generator_node,
        rqt_tf_tree_node,
        rviz_node,
    ])
