from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    share = get_package_share_directory('puzzlebot_gui')
    proxy = os.path.join(share, 'grpcwebproxy')

    return LaunchDescription([
        # Nodo ROS 2 + servidor gRPC
        Node(
            package='puzzlebot_gui',
            executable='gui_bridge',
            name='gui_bridge',
            output='screen',
            parameters=[],
        ),
        # Proxy gRPC-Web → gRPC (necesario para que el navegador pueda hablar gRPC)
        ExecuteProcess(
            cmd=[
                proxy,
                '--backend_addr=localhost:50051',
                '--run_tls_server=false',
                '--allow_all_origins',
                '--server_http_debug_port=8443',
            ],
            output='screen',
        ),
    ])
