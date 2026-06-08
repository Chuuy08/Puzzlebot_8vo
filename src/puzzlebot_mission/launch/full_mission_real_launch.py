#!/usr/bin/env python3
"""
full_mission_real_launch.py — Lanza, en el orden necesario, el stack completo
de la misión autónoma en el robot real: MCL → navegación → visión → mission_manager.

Requiere correr POR SEPARADO, en otras terminales (ciclo de vida / hardware
distinto, no tiene sentido incluirlos aquí):
  - micro_ros_agent (puente serial al microcontrolador):
      ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0
  - cámara:
      ros2 launch ros_deep_learning video_source.ros2.launch \
          input_width:=320 input_height:=180
  - fpga_controller_node (corre en la Jetson del robot, requiere acceso SPI real)

IMPORTANTE — solo UN nodo de visión a la vez: este launch usa `align_and_approach`
(el que tiene los fixes de alineación más recientes). NO corras también `tracking`
en paralelo — ambos publican a los MISMOS tópicos de control (/cmd_vel,
/alineation/booleano, /pallet_detected, ...) y competirían entre sí.
"""

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    loc_share     = get_package_share_directory('puzzlebot_localisation')
    nav_share     = get_package_share_directory('puzzlebot_navigation')
    mission_share = get_package_share_directory('puzzlebot_mission')

    # Mismo default que mcl_real_launch.py — pasa el tuyo con map:=/ruta/a/tu_mapa.yaml
    default_map = os.path.join(loc_share, 'maps', 'map_1779562705.yaml')

    map_arg = DeclareLaunchArgument(
        'map',
        default_value=default_map,
        description='Ruta absoluta al .yaml del mapa para MCL real. '
                    'Para el robot real pasa el tuyo: map:=/ruta/a/tu_mapa.yaml'
    )
    map_yaml = LaunchConfiguration('map')

    waypoints_yaml = os.path.join(mission_share, 'config', 'waypoints.yaml')

    # ── 1. MCL real ───────────────────────────────────────────────────────
    mcl = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(loc_share, 'launch', 'mcl_real_launch.py')),
        launch_arguments={
            'use_sim_time': 'false',
            'map': map_yaml,
        }.items()
    )

    # ── 2. Visión (align_and_approach) — en paralelo con MCL, carga lenta ─
    align_and_approach = Node(
        package='puzzlebot_vision',
        executable='align_and_approach',
        name='align_and_approach_node',
        output='screen',
    )

    # ── 3. Navegación real — espera a que MCL active sus nodos lifecycle ──
    navigation = TimerAction(
        period=6.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(nav_share, 'launch', 'navigation_real_launch.py')),
                launch_arguments={'use_sim_time': 'false'}.items()
            )
        ]
    )

    # ── 4. mission_manager — último: orquesta todo lo demás ───────────────
    mission_manager = TimerAction(
        period=15.0,
        actions=[
            Node(
                package='puzzlebot_mission',
                executable='mission_manager_node',
                name='mission_manager_node',
                output='screen',
                parameters=[{
                    'waypoints_yaml_path':       waypoints_yaml,
                    'sweep_range_deg':           60.0,
                    'sweep_angular_speed':       0.3,
                    'sweep_align_threshold_deg': 3.0,
                    'sweep_settle_time_s':       1.0,
                    'sweep_samples_per_stop':    5,
                    'nav_timeout_s':             90.0,
                    'fpga_settle_time_s':        2.0,
                    'delivery_inflation_radius': 0.02,
                    'delivery_robot_radius':     0.10,
                    'pallet_inflation_radius':   0.10,
                    'costmap_settle_time_s':     1.0,
                }]
            )
        ]
    )

    return LaunchDescription([
        map_arg,
        mcl,
        align_and_approach,
        navigation,
        mission_manager,
    ])
