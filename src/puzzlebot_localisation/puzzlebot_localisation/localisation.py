import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math

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

        # Subscripcione
        self.sub_wr = self.create_subscription(Float32, 'wr', self.wr_callback,10)
        self.sub_wl = self.create_subscription(Float32, 'wl', self.wl_callback,10)

        # Publicador
        self.odom_pub = self.create_publisher(Odometry, 'odom',10)

        # Transforbroadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

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

        current_time = self.get_clock().now().to_msg()

        # Publicar odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = f'{self.namespace}/base_footprint'

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

        
        