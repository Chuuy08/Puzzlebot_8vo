import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, LifecycleNode


def generate_launch_description():

    loc_pkg    = get_package_share_directory('puzzlebot_localisation')
    gazebo_pkg = get_package_share_directory('puzzlebot_gazebo')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    map_yaml     = os.path.join(loc_pkg, 'maps', 'map_1779562705.yaml')

    # ── 1. Gazebo — track world ───────────────────────────────────────────

    gazebo_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_pkg, 'launch', 'gazebo_world_launch.py')),
        launch_arguments={'world': 'track_world.sdf', 'pause': 'false'}.items()
    )

    # Spawn inside the mapped area of track_world.sdf.
    # The map (map_1779562705) covers world x∈[-1.84,4.12] y∈[-4.12,0.31].
    # (0,0,0) is the SLAM session's starting pose → guaranteed free space inside map.
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
            'k_r': 0.1592,
            'k_l': 0.2128,
            'odom_frame': 'odom',
        }]
    )

    # ── 3. Static map server ──────────────────────────────────────────────

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

    # ── 4. Custom MCL localization ────────────────────────────────────────

    mcl_node = Node(
        package='puzzlebot_localisation',
        executable='mcl_node',
        name='mcl_node',
        output='screen',
        parameters=[{
            'use_sim_time':       use_sim_time,
            'num_particles':      1500,     # más partículas → mejor mantenimiento multi-hipótesis
            'alpha1':             0.1,
            'alpha2':             0.1,
            'alpha3':             0.05,
            'alpha4':             0.05,
            'sigma_hit':          0.1,
            'sigma_hit_global':   0.20,     # bajado de 0.30 → más discriminante en búsqueda global
            'z_hit':              0.9,
            'z_rand':             0.1,
            'laser_max_range':    6.0,
            'laser_min_range':    0.15,
            'beam_step':          4,        # local: más beams para mejor tracking
            'beam_step_global':   2,        # global: máxima discriminación posible
            'update_min_d':       0.05,
            'update_min_a':       0.05,
            'laser_angle_offset': 0.0,
            'resample_interval':  1,
            'set_initial_pose':   False,
            'initial_pose_x':     0.0,
            'initial_pose_y':     0.0,
            'initial_pose_a':     0.0,
        }]
    )

    # ── 5. Localización activa — wandering hasta que MCL converja ────────────

    active_loc_node = Node(
        package='puzzlebot_localisation',
        executable='active_localization',
        name='active_localization_node',
        output='screen',
        parameters=[{
            'use_sim_time':          use_sim_time,
            'scan_topic':            '/scan',
            'wander_timeout':        120.0,
            'wander_linear_speed':   0.10,
            'wander_angular_speed':  0.50,
            'obstacle_distance':     0.40,
            'random_turn_interval':   5.0,
            'convergence_hold_time':  5.0,
            'min_conv_travel_m':      0.20,
            'min_conv_rotation_deg':  90.0,
            'front_half_angle_deg':   60.0,
        }]
    )

    # ── 6. RViz ───────────────────────────────────────────────────────────
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
        DeclareLaunchArgument('use_sim_time', default_value='true'),

        gazebo_world,
        spawn_robot,
        localisation_node,
        map_server,
        lifecycle_manager,
        mcl_node,
        active_loc_node,
        rviz_node,
    ])