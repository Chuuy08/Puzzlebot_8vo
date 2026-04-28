# Modelo cinemático (mundo continuo)
# x_punto = v * cos(theta)
# y_punto = v * sin(theta)
# theta_punto = w

# Relacion ruedas <-> robot
# v = r(wR + wL) / 2
# w = r(wR - wL) / l

# Debemos pasar de continuo a discreto
# Integracion 
# sx <- sx + v * cos(theta) * dt
# sy <- sy + v * sin(theta) * dt
# stheta <- stheta + w * dt


# Despejamos v y w de la relacion ruedas <-> robot
# wR = (v + w * l / 2) / r
# wL = (v - w * l / 2) / r

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float32
import math


class PuzzleBotKinematic(Node):
    def __init__(self):
        super().__init__('puzzlebot_kinematic')

        # Parametros configurables dede launch
        
        self.declare_parameter('wheel_radius', 0.05) # r [meters]
        self.declare_parameter('wheel_base', 0.19) # l [meters]
        self.declare_parameter('sampling_time', 0.05) # dt [s]
        self.declare_parameter('x0', 0.0)
        self.declare_parameter('y0', 0.0)
        self.declare_parameter('theta0', 0.0)
        
        self.r = self.get_parameter('wheel_radius').value
        self.l = self.get_parameter('wheel_base').value
        self.dt = self.get_parameter('sampling_time').value

        # Estado del robot 
        self.sx = self.get_parameter('x0').value
        self.sy = self.get_parameter('y0').value
        self.stheta = self.get_parameter('theta0').value

        # Entrada del robot 
        self.v = 0.0 # velocidad lineal
        self.w = 0.0 # velocidad angular

        # Subscripciones
        self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        self.pose_pub = self.create_publisher(PoseStamped, 'pose_sim', 10)
        self.wr_pub = self.create_publisher(Float32, 'wr', 10)
        self.wl_pub = self.create_publisher(Float32, 'wl', 10) 

        # Timer
        self.create_timer(self.dt, self.timer_callback)
        self.get_logger().info(f'Kinematic model listo | r={self.r} m | l={self.l} m | dt={self.dt} s')

    def cmd_vel_callback(self, msg: Twist):
        self.v = msg.linear.x
        self.w = msg.angular.z
    
    def timer_callback(self):
        # velocidad de cada rueda
        wr = (self.v + self.w * self.l / 2.0) / self.r
        wl = (self.v - self.w * self.l / 2.0) / self.r

        # Integracion de euler
        self.sx += self.v * math.cos(self.stheta) * self.dt
        self.sy += self.v * math.sin(self.stheta) * self.dt
        self.stheta += self.w * self.dt

        # Normalizar angulo
        self.stheta = math.atan2(math.sin(self.stheta), math.cos(self.stheta))  

        #  Cuaternion 
        qz = math.sin(self.stheta / 2)
        qw = math.cos(self.stheta / 2)

        # Publicar pose 
        pose_msg = PoseStamped()        
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'odom'
        pose_msg.pose.position.x = self.sx
        pose_msg.pose.position.y = self.sy
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        self.pose_pub.publish(pose_msg) 

        # Publicar velocidades
        wr_msg = Float32()
        wl_msg = Float32()
        wr_msg.data = wr
        wl_msg.data = wl
        self.wr_pub.publish(wr_msg)
        self.wl_pub.publish(wl_msg) 

        # Log
        #self.get_logger().info(f'Pose: ({self.sx:.2f}, {self.sy:.2f}, {self.stheta:.2f}) | V={self.v:.2f} m/s | W={self.w:.2f} rad/s')        

def main(args=None):
    rclpy.init(args=args)
    node = PuzzleBotKinematic()
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
        
