import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math
import numpy as np
from .utils import wrap_angle

class Localisation(Node):
    def __init__(self):
        super().__init__('localisation')

        # Detectar namespace automáticamente
        self.namespace = self.get_namespace().strip('/')

        # Parametros
        self.declare_parameter('wheel_radius',  0.05)
        self.declare_parameter('wheel_base',    0.19)
        self.declare_parameter('sampling_time', 0.05)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('k_r', 0.1592)
        self.declare_parameter('k_l', 0.2128)

        self.odom_frame = self.get_parameter('odom_frame').get_parameter_value().string_value.strip('/')
        # child_frame_id without leading slash — required by tf2/Cartographer
        ns = self.namespace
        self.base_frame_id = f'{ns}/base_footprint' if ns else 'base_footprint'


        self.r  = self.get_parameter('wheel_radius').value
        self.l  = self.get_parameter('wheel_base').value
        self.dt = self.get_parameter('sampling_time').value

        # Estado del robot 
        self.sx = 0.0
        self.sy = 0.0
        self.stheta = 0.0

        # velocidade del robot
        self.wr = 0.0
        self.wl = 0.0

        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.sub_wr = self.create_subscription(Float32, 'wr', self.wr_callback, best_effort_qos)
        self.sub_wl = self.create_subscription(Float32, 'wl', self.wl_callback, best_effort_qos)

        # Publicador
        self.odom_pub = self.create_publisher(Odometry, 'odom',10)

        # Transforbroadcaster
        self.tf_broadcaster = TransformBroadcaster(self)



        self.k_r = self.get_parameter('k_r').value
        self.k_l = self.get_parameter('k_l').value

        self.Sigma = np.zeros((3, 3))

        # Timer 
        self.timer = self.create_timer(self.dt, self.timer_callback)

        self.get_logger().info( f'Localisation listo | r={self.r} m | l={self.l} m | dt={self.dt} s')

    # Callbacks de /wr y /wl
    def wr_callback(self, msg: Float32):
        self.wr = msg.data

    def wl_callback(self, msg: Float32):
        self.wl = msg.data

    # Dead reckoning
    def timer_callback(self):
        # Velocidad lineal y angular
        v = self.r*(self.wr + self.wl) / 2.0 
        w = self.r*(self.wr - self.wl) / self.l
        # Desplazamiento incremental  
        delta_d = v * self.dt
        delta_theta = w * self.dt

        # Propagación de covarianza ANTES de integrar el estado: Ak y Jw se
        # evalúan en θ_k, no θ_{k+1}.

        # Ak: jacobiana de la transición respecto al estado
        Ak = np.array([
            [1, 0, -v * self.dt * math.sin(self.stheta)],
            [0, 1,  v * self.dt * math.cos(self.stheta)],
            [0, 0, 1]
        ]) 

        # Encoder noise covariance — proportional to wheel speed magnitude
        Sigma_delta = np.array([
            [self.k_r * abs(self.wr), 0.0],
            [0.0,                     self.k_l * abs(self.wl)]
        ])

        # Jacobian of the motion model w.r.t. wheel velocities (wr, wl)
        Jw = np.array([
            [(self.r * self.dt / 2.0) * math.cos(self.stheta), (self.r * self.dt / 2.0) * math.cos(self.stheta)],
            [(self.r * self.dt / 2.0) * math.sin(self.stheta), (self.r * self.dt / 2.0) * math.sin(self.stheta)],
            [ self.r * self.dt / self.l,                       -(self.r * self.dt / self.l)]
        ])

        # Q_k = Jw @ Sigma_delta @ Jw.T
        Qk = Jw @ Sigma_delta @ Jw.T

        # Propagacion de la incertidumbre (Σ grows each step)
        self.Sigma = Ak @ self.Sigma @ Ak.T + Qk

        # --- State integration (pose update) ---
        self.sx += delta_d * math.cos(self.stheta) 
        self.sy += delta_d * math.sin(self.stheta)
        self.stheta += delta_theta
        
        self.stheta = wrap_angle(self.stheta)

        # Quaternion
        qz = math.sin(self.stheta / 2.0)
        qw = math.cos(self.stheta / 2.0)

        current_time = self.get_clock().now().to_msg()

        # Publicar odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame_id

        # Mapear Sigma 3x3 al bloque x/y/yaw de la covarianza 6x6 de ROS (resto en cero)
        cov = [0.0] * 36
        cov[0]  = self.Sigma[0, 0]  # x-x
        cov[1]  = self.Sigma[0, 1]  # x-y
        cov[5]  = self.Sigma[0, 2]  # x-yaw
        cov[6]  = self.Sigma[1, 0]  # y-x
        cov[7]  = self.Sigma[1, 1]  # y-y
        cov[11] = self.Sigma[1, 2]  # y-yaw
        cov[30] = self.Sigma[2, 0]  # yaw-x
        cov[31] = self.Sigma[2, 1]  # yaw-y
        cov[35] = self.Sigma[2, 2]  # yaw-yaw (theta variance)
        odom_msg.pose.covariance = cov

        # Pose
        odom_msg.pose.pose.position.x = self.sx
        odom_msg.pose.pose.position.y = self.sy
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation.x = 0.0
        odom_msg.pose.pose.orientation.y = 0.0
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw

        # Twist - velocidades actuales en el frame del robot
        odom_msg.twist.twist.linear.x = v
        odom_msg.twist.twist.angular.z = w


        self.odom_pub.publish(odom_msg)

        # Publicar el tf
        tf_msg = TransformStamped()
        tf_msg.header.stamp = current_time
        tf_msg.header.frame_id = self.odom_frame
        tf_msg.child_frame_id = self.base_frame_id


        tf_msg.transform.translation.x = self.sx
        tf_msg.transform.translation.y = self.sy
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.x = 0.0
        tf_msg.transform.rotation.y = 0.0
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(tf_msg)    

def main(args=None):
    rclpy.init(args=args)
    node = Localisation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()  

        
        