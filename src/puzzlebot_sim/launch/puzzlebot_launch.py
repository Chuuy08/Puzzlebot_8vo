from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    static_transform_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['2', '1', '0', '0', '0', '0', 'map', 'odom'] 
    )

    puzzlebot_transform_node = Node(
        name='puzzlebotcar',
        package='puzzlebot_sim',
        executable='PuzzlebotCar',
    )

    rviz_config = os.path.join(
        get_package_share_directory('puzzlebot_sim'),
        'rviz',
        'config.rviz'
    )
    rviz_node = Node(
        name='rviz',
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config]
    )

    rqt_tf_tree_node = Node(
        name='rqt_tf_tree',
        package='rqt_tf_tree',
        executable='rqt_tf_tree',
    )

    l_d = LaunchDescription([static_transform_node, puzzlebot_transform_node, rviz_node, rqt_tf_tree_node])
    return l_d