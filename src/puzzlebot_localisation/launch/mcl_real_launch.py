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
    map_yaml     = LaunchConfiguration('map')

    urdf_path  = os.path.join(desc_pkg, 'urdf', 'mcr2_robots', 'puzzlebot_jetson_lidar_ed.xacro')
    robot_desc = ParameterValue(Command(['xacro ', urdf_path]), value_type=str)

    # ── 1. TF tree ────────────────────────────────────────────────────────

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

    # sllidar publishes frame_id="laser"; URDF names the link "laser_frame"
    laser_frame_bridge = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='laser_frame_bridge',
        arguments=['0', '0', '0', '3.14159', '0', '0', 'laser_frame', 'laser'],
    )

    # ── 2. Dead-reckoning odometry ────────────────────────────────────────

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
            'k_r':           0.1592,
            'k_l':           0.2128,
            'odom_frame':    'odom',
        }],
        remappings=[
            ('wr', 'VelocityEncR'),
            ('wl', 'VelocityEncL'),
        ]
    )

    # ── 3. Scan relay: fix sllidar hardware timestamps ───────────────────────

    scan_relay = Node(
        package='puzzlebot_localisation',
        executable='scan_relay',
        name='scan_relay',
        output='screen',
        remappings=[
            ('scan_raw', '/scan'),       # read from sllidar
            ('scan',     '/scan_fixed'), # publish with ROS clock timestamp
        ],
    )

    # ── 4. Static map server ──────────────────────────────────────────────

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

    # ── 5. MCL localization ───────────────────────────────────────────────
    # Scan is remapped to the timestamp-corrected topic.
    # Real-robot params: slightly more noise tolerance than simulation.

    mcl_node = Node(
        package='puzzlebot_localisation',
        executable='mcl_node',
        name='mcl_node',
        output='screen',
        parameters=[{
            'use_sim_time':        use_sim_time,
            'num_particles':       2800,
            'alpha1':              0.1,
            'alpha2':              0.1,
            'alpha3':              0.05,
            'alpha4':              0.05,
            'sigma_hit':           0.12,   # tight — used when converged/tracking
            'sigma_hit_global':    0.30,   # loose — used during global search
            'z_hit':               0.85,
            'z_rand':              0.15,
            'laser_max_range':     4.5,
            'laser_min_range':     0.15,
            'beam_step':           8,      # fewer beams when tracking (fast)
            'beam_step_global':    3,      # more beams during global search
            'update_min_d':        0.05,
            'update_min_a':        0.12,   # ~7° — avoids noise-triggered updates
            'resample_interval':   1,
            
            'laser_angle_offset':  3.14159,  # π — RPLidar cable faces rear of robot

            'set_initial_pose':    False,
            'initial_pose_x':      0.0,
            'initial_pose_y':      0.0,
            'initial_pose_a':      0.0,
            'init_pose_spread_xy': 0.40,   # ±0.4m around clicked position
            'init_pose_spread_a':  0.30,   # ±17° around RViz arrow direction
        }],
        remappings=[('/scan', '/scan_fixed')]
    )

    # ── 6. Localización activa — wandering hasta que MCL converja ────────────
    # scan_topic apunta al scan con timestamps corregidos, igual que mcl_node.
    # Velocidades conservadoras para el hardware real.

    active_loc_node = Node(
        package='puzzlebot_localisation',
        executable='active_localization',
        name='active_localization_node',
        output='screen',
        parameters=[{
            'use_sim_time':          use_sim_time,
            'scan_topic':            '/scan_fixed',
            'wander_timeout':        120.0,
            'wander_linear_speed':   0.08,
            'wander_angular_speed':  0.40,
            'obstacle_distance':     0.40,
            'random_turn_interval':   5.0,
            'convergence_hold_time':  2.0,
            'min_conv_travel_m':      0.05,
            'min_conv_rotation_deg':  90.0,
            'front_half_angle_deg':   60.0,
        }]
    )

    # ── 7. RViz ───────────────────────────────────────────────────────────
    # Strip snap's libpthread from LD_LIBRARY_PATH to avoid the GLIBC_PRIVATE error
    ld_lib = os.environ.get('LD_LIBRARY_PATH', '')
    ld_lib_clean = ':'.join(p for p in ld_lib.split(':') if 'snap' not in p)

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        additional_env={'LD_LIBRARY_PATH': ld_lib_clean},
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'map',
            default_value=os.path.join(loc_pkg, 'maps', 'map_1779562705.yaml'),
            description='Absolute path to the map YAML. For the real robot pass your own map: '
                        'map:=/path/to/your_map.yaml',
        ),

        robot_state_pub,
        joint_state_pub,
        laser_frame_bridge,
        localisation_node,
        scan_relay,
        map_server,
        lifecycle_manager,
        mcl_node,
        active_loc_node,
        rviz_node,
    ])
