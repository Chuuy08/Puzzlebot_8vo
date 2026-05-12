import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math

import numpy as np

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

        self.odom_frame = self.get_parameter('odom_frame').get_parameter_value().string_value.strip('/')


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



        # Variable covarianza
        self.Sigma = np.zeros((3,3))

        # --- PROBABILISTIC LOCALISATION ADDITION: Noise coefficients ---
        # Encoder noise proportionality constants — tune these to match real hardware
        self.k_r = 0.05  # right wheel noise coefficient
        self.k_l = 0.05  # left wheel noise coefficient

        # Timer 
        self.timer = self.create_timer(self.dt, self.timer_callback)

        self.get_logger().info( f'Localisation listo | r={self.r} m | l={self.l} m | dt={self.dt} s')

    # Callbacks de /wr y /wl
    def wr_callback(self, msg: Float32):
        self.get_logger().info(f"{msg.data}")
        self.wr = msg.data

    def wl_callback(self, msg: Float32):
        self.get_logger().info(f"{msg.data}")
        self.wl = msg.data 

    # Dead reckoning
    def timer_callback(self):
        # Velocidad lineal y angular
        v = self.r*(self.wr + self.wl) / 2.0 
        w = self.r*(self.wr - self.wl) / self.l
        # Desplazamiento incremtanl  
        delta_d = v * self.dt
        delta_theta = w * self.dt

        # Integracion de euler (pose)
        self.sx += delta_d * math.cos(self.stheta) 
        self.sy += delta_d * math.sin(self.stheta)
        self.stheta += delta_theta
        
        # normalizar angulo
        self.stheta = math.atan2(math.sin(self.stheta), math.cos(self.stheta))

        # Quaternion
        qz = math.sin(self.stheta / 2.0)
        qw = math.cos(self.stheta / 2.0)   

        # Ak es la jacobiana de la función de transición respecto al estado, necesaria para propagar la covarianza
        Ak = np.array([
            [1, 0, -v * self.dt * math.sin(self.stheta)],
            [0, 1,  v * self.dt * math.cos(self.stheta)],
            [0, 0, 1]
        ]) 



        # --- PROBABILISTIC LOCALISATION ADDITION: Realistic process noise Q_k ---
        # Step 1: encoder noise covariance — proportional to wheel speed magnitude
        # Faster wheels → larger uncertainty injected per step
        Sigma_delta = np.array([
            [self.k_r * abs(self.wr), 0.0],
            [0.0,                     self.k_l * abs(self.wl)]
        ])

        # Step 2: Jacobian of the motion model w.r.t. wheel velocities (wr, wl)
        # Maps wheel-speed uncertainty into (x, y, theta) state space
        Jw = np.array([
            [(self.r * self.dt / 2.0) * math.cos(self.stheta), (self.r * self.dt / 2.0) * math.cos(self.stheta)],
            [(self.r * self.dt / 2.0) * math.sin(self.stheta), (self.r * self.dt / 2.0) * math.sin(self.stheta)],
            [ self.r * self.dt / self.l,                       -(self.r * self.dt / self.l)]
        ])

        # Step 3: Q_k = Jw @ Sigma_delta @ Jw.T
        # Covariance grows with motion — robot standing still injects almost no uncertainty
        Qk = Jw @ Sigma_delta @ Jw.T

        # Propagacion de la incertidumbre
        # Sigma grows each step: Ak spreads existing uncertainty, Qk adds new motion noise
        self.Sigma = Ak @ self.Sigma @ Ak.T + Qk

        # --- PROBABILISTIC LOCALISATION ADDITION: Debug print ---
        print(self.Sigma)

        current_time = self.get_clock().now().to_msg()

        # Publicar odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = f'{self.namespace}/base_footprint'

        # --- PROBABILISTIC LOCALISATION ADDITION: Map 3x3 Sigma into 6x6 ROS covariance ---
        # 6x6 row-major for [x, y, z, roll, pitch, yaw]
        # Only the x/y/yaw subblock is populated; remaining DOFs are zero
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
        tf_msg.child_frame_id = f'{self.namespace}/base_footprint'


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

        
        