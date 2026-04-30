import os
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions.path_join_substitution import PathJoinSubstitution

ARGUMENTS = [
    DeclareLaunchArgument('use_sim_time', default_value='true', choices=['true', 'false'], description='Use sim time'),
]

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    pkg_description = get_package_share_directory('puzzlebot_description')
    pkg_ros_ign_gazebo = get_package_share_directory('ros_gz_sim')

    # GZ_SIM_RESOURCE_PATH para que encuentre models/map/
    ign_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=[os.path.join(pkg_description, 'models') + ':$GZ_SIM_RESOURCE_PATH']
    )

    # Ruta al world.sdf
    world_sdf_path = os.path.join(pkg_description, 'models', 'worlds', 'world.sdf')

    ign_gazebo_launch = PathJoinSubstitution([pkg_ros_ign_gazebo, 'launch', 'gz_sim.launch.py'])

    ignition_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([ign_gazebo_launch]),
        launch_arguments={
            'gz_args': world_sdf_path + ' -r -v 4'
        }.items()
    )

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_bridge',
        output='screen',
        arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock']
    )

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_odom',
        output='screen',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--roll', '0', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'map', '--child-frame-id', 'odom'],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(pkg_description, 'rviz', 'map.rviz')],
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    map_publisher = Node(
    package='puzzlebot_description',
    executable='map_pub',
    name='map_pub',
    output='screen',
    parameters=[{'use_sim_time': use_sim_time}]
    )

    return LaunchDescription([
        *ARGUMENTS,
        ign_resource_path,
        ignition_gazebo,
        clock_bridge,
        static_tf,
        rviz_node,
        map_publisher,
    ])