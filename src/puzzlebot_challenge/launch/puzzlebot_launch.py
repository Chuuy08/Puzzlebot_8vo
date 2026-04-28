import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    # Parametros centralizados
    robot_params = {
        'wheel_radius':  0.05,   
        'wheel_base':    0.19,    
        'sampling_time': 0.05,   
    }

    # Cargar archivo URDF
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    urdf_file_name = 'model.urdf'
    urdf = os.path.join(
        get_package_share_directory('puzzlebot_challenge'),
        'urdf',
        urdf_file_name)
        
    with open(urdf, 'r') as infp:
        robot_desc = infp.read()

    # Nodo 1: puzzlebot_kinematics
    kinematic_model_node = Node(
        package='puzzlebot_challenge',
        executable='puzzlebot_kinematic',
        name='puzzlebot_kinematic',
        output='screen',
        parameters=[robot_params]   
    )

    # Nodo 2: localisation
    localisation_node = Node(
        package='puzzlebot_challenge',
        executable='localisation',
        name='localisation',
        output='screen',
        parameters=[robot_params] 
    )

    # Nodo 3: Joint_state_publisher
    joint_state_node = Node(
        package='puzzlebot_challenge',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        parameters=[{'sampling_time': robot_params['sampling_time']}]
    )

    # NOdo 4: Robot_state_publisher
    robot_state_pub_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time, 'robot_description': robot_desc}]
    )

    # Nodo 5 : Control
    control_node = Node(
        package='puzzlebot_challenge',
        executable='control',
        name='control',
        output='screen',
        parameters=[{
        # 'x_goal': 2.0,
        # 'y_goal': 0.0,
        'Kd': 0.25,
        'Ktheta': 1.8,
        'threshold': 0.1,
        'sampling_time': 0.05,
    }]  
    )

    # Nodo 6: Setpoint Generator
    setpoint_generator_node = Node(
        package='puzzlebot_challenge',
        executable='set_poin_generator',
        name='setpoint_generator',
        output='screen',
        parameters=[{
            'trajectory': 'square',
            'side_length': 1.0,
            'loop': False,
        }]
    )


    # Transformadas estaticas
    static_transform_node_map_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments = ['--x', '0', '--y', '0', '--z', '0',
                    '--yaw', '0', '--pitch', '0', '--roll', '0',
                    '--frame-id', 'map', '--child-frame-id', 'odom']
    )

    static_transform_node_world_map = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments = ['--x', '0', '--y', '0', '--z', '0',
                    '--yaw', '0', '--pitch', '0', '--roll', '0',
                    '--frame-id', 'world', '--child-frame-id', 'map']
    )

    # Teleop twist 
    teleop_node = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        name='teleop_twist_keyboard',
        output='screen',
        prefix='xterm -e',    # <- string, no lista
    )

    # Rqt TF Tree
    rqt_tf_tree_node = Node(name='rqt_tf_tree',
        package='rqt_tf_tree',
        executable='rqt_tf_tree'
    )

    # Rviz
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
        #teleop_node,
        rqt_tf_tree_node,
        rviz_node,
    ])


"""
Hola cat buenos dias, ocupo que me ayudes a aclarar algunas ideas Estoy estudiando la carrera de ingenieira en robotica y sistemas digitales. Ahorita me encuentro en 8vo semestre y estoy viendo el uso del lidar y SLAM. Por lo que para comprender de mejor manera esto, nos pideiron intenar lealizar nestro propio pipeline de SLAM y usando tambien lo del MOntecarlo. Pero estoy perdido ya que no se por donde empezar. Iggual nos dijeron que crearamos nuestro propio mapa y ke definieramos las medidas para poder ralizar lo que nos soliciatron esots son como qu elas actiidades que vimos: 1. ¿Qué es un mapa? 2. LIDAR 3. Simultaneous Localization and Mapping (SLAM) 4. Localización 2D con Montecarlo 5. Actividad Localización 2D 6. Mapeo 2D 7. Actividad Mapeo 2D Y esto como que nos pideiron, dime como empezar y que empxzar a investigar o como me ayudarias tu para complrender lo que me soliciatn Intentar Montecarlo Localization • A. Utilicen un simulador simple como CoppeliaSim o Gazebo. • B. Generen un layout de su entorno (se supone que el mapa ya es conocido). • C. Decidan las dimensiones del grid (relación metros/pixel). • D. Hagan un muestreo de partículas. • E. Asignen puntajes a cada partícula (basado en sumas de valores de pixeles). • F. Filtren las partículas (quédense sólo con las de mayor puntaje). • G. Estimen el avance del robot para la siguiente iteración con Dead Reckoning. • H. Muevan todos sus mejores candidatos de partículas en la dirección del robot. • I. Repitan desde el paso D.

"""