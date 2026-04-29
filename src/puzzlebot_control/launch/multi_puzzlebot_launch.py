import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # Parametros ambos robots
    robot_params = {
        'wheel_radius':  0.05,
        'wheel_base':    0.19,
        'sampling_time': 0.05,
    }

    # URDF
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    urdf_path = os.path.join(
        get_package_share_directory('puzzlebot_description'),
        'urdf',
        'model.urdf'
    )
    with open(urdf_path, 'r') as f:
        robot_desc_base = f.read()


    # TFs estáticos globales
    static_tf_world_map = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_world_map',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--yaw', '0', '--pitch', '0', '--roll', '0',
                   '--frame-id', 'world', '--child-frame-id', 'map']
    )

    static_tf_map_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_odom',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--yaw', '0', '--pitch', '0', '--roll', '0',
                   '--frame-id', 'map', '--child-frame-id', 'odom']
    )

    r1_robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace='robot1',
        output='screen',
        parameters=[{
            'use_sim_time':      use_sim_time,
            'robot_description': robot_desc_base,
            'frame_prefix':            'robot1/',
        }]
    )

    # Robot 1
    r1_kinematic = Node(
        package='puzzlebot_challenge',
        executable='puzzlebot_kinematic',
        name='puzzlebot_kinematic',
        namespace='robot1',
        output='screen',
        parameters=[robot_params]
    )

    r1_localisation = Node(
        package='puzzlebot_localisation',
        executable='localisation',
        name='localisation',
        namespace='robot1',
        output='screen',
        parameters=[{**robot_params, 'odom_frame': 'odom'}]
    )

    r1_joint_state = Node(
        package='puzzlebot_description',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        namespace='robot1',
        output='screen',
        parameters=[{'sampling_time': robot_params['sampling_time']}]
    )

    r1_control = Node(
        package='puzzlebot_control',
        executable='control',
        name='control',
        namespace='robot1',
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
            'v_max': 0.2, 
            'w_max': 1.0,
        }]
    )

    r1_setpoint = Node(
        package='puzzlebot_control',
        executable='set_poin_generator',
        name='set_poin_generator',
        namespace='robot1',
        output='screen',
        parameters=[{'trajectory': 'square', 'side_length': 1.0, 'loop': False}]
    )

    r2_robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace='robot2',
        output='screen',
        parameters=[{
            'use_sim_time':      use_sim_time,
            'robot_description': robot_desc_base,
            'frame_prefix':            'robot2/',
        }]
    )

    # Robot 2
    r2_kinematic = Node(
        package='puzzlebot_challenge',
        executable='puzzlebot_kinematic',
        name='puzzlebot_kinematic',
        namespace='robot2',
        output='screen',
        parameters=[{**robot_params, 'x0': 0.0, 'y0': 1.5}]
    )

    r2_localisation = Node(
        package='puzzlebot_localisation',
        executable='localisation',
        name='localisation',
        namespace='robot2',
        output='screen',
        parameters=[{**robot_params, 'odom_frame': 'odom'}]
    )

    r2_joint_state = Node(
        package='puzzlebot_description',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        namespace='robot2',
        output='screen',
        parameters=[{'sampling_time': robot_params['sampling_time']}]
    )   

    r2_control = Node(
        package='puzzlebot_control',
        executable='control',
        name='control',
        namespace='robot2',
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
            'v_max': 0.2, 
            'w_max': 1.0,
        }]
    )

    r2_setpoint = Node(
        package='puzzlebot_control',
        executable='set_poin_generator',
        name='set_poin_generator',
        namespace='robot2',
        output='screen',
        parameters=[{'trajectory': 'pentagon', 'side_length': 1.0, 'loop': False}]
    )

    rviz_config = os.path.join(
        get_package_share_directory('puzzlebot_challenge'),
        'rviz',
        'two_robots.rviz'
    )

    # Visualización
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
    )

    rqt_tf_tree_node = Node(
        name='rqt_tf_tree',
        package='rqt_tf_tree',
        executable='rqt_tf_tree'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true'),

        # TFs globales
        static_tf_world_map,
        static_tf_map_odom,

        # Robot 1
        r1_robot_state_pub,
        r1_kinematic,
        r1_localisation,
        r1_joint_state,
        r1_control,
        r1_setpoint,

        # Robot 2
        r2_robot_state_pub,
        r2_kinematic,
        r2_localisation,
        r2_joint_state,
        r2_control,
        r2_setpoint,

        # Visualización
        rqt_tf_tree_node,
        rviz_node,
    ])