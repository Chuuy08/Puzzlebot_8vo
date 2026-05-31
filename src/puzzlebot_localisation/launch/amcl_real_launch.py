import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node, LifecycleNode
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    loc_pkg  = get_package_share_directory('puzzlebot_localisation')
    desc_pkg = get_package_share_directory('puzzlebot_description')

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    map_yaml = os.path.join(loc_pkg, 'maps', 'map_1779562705.yaml')

    urdf_path  = os.path.join(desc_pkg, 'urdf', 'mcr2_robots', 'puzzlebot_jetson_lidar_ed.xacro')
    robot_desc = ParameterValue(Command(['xacro ', urdf_path]), value_type=str)

    # ── Robot description & TF tree ───────────────────────────────────────

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

    # RPLidar publishes frame_id="laser"; URDF names the link "laser_frame"
    laser_frame_bridge = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='laser_frame_bridge',
        arguments=['0', '0', '0', '0', '0', '0', 'laser_frame', 'laser'],
    )

    # ── Dead-reckoning odometry (odom → base_footprint TF + /odom) ───────

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

    # ── Map server (serves static map on /map) ────────────────────────────
    # nav2_map_server is a lifecycle node; lifecycle_manager activates it.

    map_server = LifecycleNode(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        namespace='',
        output='screen',
        parameters=[{
            'use_sim_time':  use_sim_time,
            'yaml_filename': map_yaml,
        }]
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart':    True,
            'node_names':   ['map_server'],
        }]
    )

    # ── Custom MCL localization node ──────────────────────────────────────
    # Subscribes to /map, /odom, /scan, /initialpose
    # Publishes: mcl_pose, particle_cloud, TF map→odom

    mcl_node = Node(
        package='puzzlebot_localisation',
        executable='mcl_node',
        name='mcl_node',
        output='screen',
        parameters=[{
            'use_sim_time':        use_sim_time,
            'num_particles':       500,
            'alpha1':              0.2,
            'alpha2':              0.2,
            'alpha3':              0.1,
            'alpha4':              0.1,
            'sigma_hit':           0.2,
            'z_hit':               0.8,
            'z_rand':              0.2,
            'laser_max_range':     6.0,
            'laser_min_range':     0.15,
            'beam_step':           10,
            'update_min_d':        0.10,
            'update_min_a':        0.10,
            'resample_interval':   2,
            'laser_angle_offset':  3.14159,  # π — RPLidar cable faces rear of robot
            'set_initial_pose':    False,
            'initial_pose_x':      0.0,
            'initial_pose_y':      0.0,
            'initial_pose_a':      0.0,
        }],
        remappings=[('/scan', '/scan_fixed')]
    )

    # ── Visualisation ─────────────────────────────────────────────────────

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
        map_server,
        lifecycle_manager,
        mcl_node,
        rviz_node,
    ])
