"""
navigation_launch.py — Navegación autónoma completa

  # Robot real (cambiar scan_topic y laser_angle_offset):
  ros2 launch puzzlebot_navigation navigation_launch.py \\
      scan_topic:=/scan_fixed laser_angle_offset:=3.14159
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Argumentos configurables desde la terminal ────────────────────────
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='true para simulación, false para robot real')

    scan_topic_arg = DeclareLaunchArgument(
        'scan_topic', default_value='/scan',
        description='/scan (sim) o /scan_fixed (robot real)')

    laser_angle_offset_arg = DeclareLaunchArgument(
        'laser_angle_offset', default_value='0.0',
        description='0.0 (sim) o 3.14159 (robot real, cable RPLidar atrás)')

    inflation_radius_arg = DeclareLaunchArgument(
        'inflation_radius', default_value='0.18',
        description='Radio de inflado estático del costmap [m]')

    robot_radius_arg = DeclareLaunchArgument(
        'robot_radius', default_value='0.13',
        description='Radio del robot en DWA — debe ser < inflation_radius')

    # ── Variables de configuración ────────────────────────────────────────
    use_sim_time      = LaunchConfiguration('use_sim_time')
    scan_topic        = LaunchConfiguration('scan_topic')
    laser_offset      = LaunchConfiguration('laser_angle_offset')
    inflation_radius  = LaunchConfiguration('inflation_radius')
    robot_radius      = LaunchConfiguration('robot_radius')

    # ── 1. Costmap: mapa estático inflado + capa dinámica del LiDAR ──────
    costmap_node = Node(
        package='puzzlebot_navigation',
        executable='costmap_node',
        name='costmap_node',
        output='screen',
        parameters=[{
            'use_sim_time':             use_sim_time,
            'scan_topic':               scan_topic,
            'laser_angle_offset':       laser_offset,
            'inflation_radius':         inflation_radius,
            'dynamic_inflation_radius': inflation_radius,
            'publish_rate':             5.0,
            'laser_x_offset':           0.0425,
            'laser_y_offset':           0.0,
            'laser_max_range':          4.5,
            'laser_min_range':          0.15,
        }]
    )

    # ── 2. RRT: planificador global bidireccional ─────────────────────────
    rrt_node = Node(
        package='puzzlebot_navigation',
        executable='rrt_node',
        name='rrt_node',
        output='screen',
        parameters=[{
            'use_sim_time':  use_sim_time,
            'step_size':     0.20,
            'max_iterations': 6000,
            'goal_bias':     0.15,
            'smooth_path':   True,
        }]
    )

    # ── 3. Path follower: Pure Pursuit sobre el path global ───────────────
    path_follower_node = Node(
        package='puzzlebot_navigation',
        executable='path_follower_node',
        name='path_follower_node',
        output='screen',
        parameters=[{
            'use_sim_time':        use_sim_time,
            'lookahead_distance':  0.40,
            'linear_speed':        0.18,
            'max_angular_speed':   1.20,
            'goal_tolerance':      0.15,
            'align_threshold_deg': 35.0,
            'drive_threshold_deg':  8.0,
            'speed_curve_gain':    0.50,
            'control_rate':       20.0,
            'output_topic':       '/cmd_vel_reference',
        }]
    )

    # ── 4. DWA: control local con evasión de obstáculos ───────────────────
    dwa_node = Node(
        package='puzzlebot_navigation',
        executable='dwa_node',
        name='dwa_node',
        output='screen',
        parameters=[{
            'use_sim_time':   use_sim_time,
            'scan_topic':     scan_topic,
            'v_max':          0.20,
            'omega_max':      1.20,
            'accel_v':        0.80,
            'accel_omega':    1.60,
            'robot_radius':   robot_radius,
            'sim_time':       1.0,
            'sim_dt':         0.10,
            'v_samples':      8,
            'omega_samples':  16,
            'w_heading':      0.50,
            'w_clearance':    0.30,
            'w_velocity':     0.20,
            'lookahead':      0.50,
            'goal_tolerance': 0.15,
            'control_rate':  20.0,
        }]
    )

    return LaunchDescription([
        use_sim_time_arg,
        scan_topic_arg,
        laser_angle_offset_arg,
        inflation_radius_arg,
        robot_radius_arg,
        costmap_node,
        rrt_node,
        path_follower_node,
        dwa_node,
    ])
