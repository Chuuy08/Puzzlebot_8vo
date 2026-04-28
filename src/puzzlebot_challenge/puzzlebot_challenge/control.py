import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist,Vector3
from nav_msgs.msg import Odometry
import math
from std_msgs.msg import Bool


def wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

class Control(Node):
    def __init__(self):
        super().__init__('control')
        # self.declare_parameter('x_goal', 2.0)
        # self.declare_parameter('y_goal',0.0)
        self.declare_parameter('Kd',0.3)
        self.declare_parameter('Ktheta', 0.6)
        self.declare_parameter('threshold', 0.1) # 10 cm
        self.declare_parameter('sampling_time', 0.05)
        self.declare_parameter('v_max', 0.2)
        self.declare_parameter('w_max', 1.0)

        # self.x_goal = self.get_parameter('x_goal').value
        # self.y_goal = self.get_parameter('y_goal').value
        self.Kd = self.get_parameter('Kd').value
        self.Ktheta = self.get_parameter('Ktheta').value
        self.threshold = self.get_parameter('threshold').value
        self.dt = self.get_parameter('sampling_time').value
        self.v_max = self.get_parameter('v_max').value
        self.w_max = self.get_parameter('w_max').value

        # Pose actual del robot
        self.xr = 0.0
        self.yr = 0.0
        self.thetar = 0.0

        # Flag para saber si ya llegamos 
        self.x_goal = None
        self.y_goal = None

        self.goal_reached = False 
        self._log_count = 0 

        # Subscripcion 
        self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.create_subscription(Vector3,'set_point',self.set_point_callback,10)
        
        # Publicador 
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.goal_reached_pub = self.create_publisher(Bool, 'goal_reached', 10)   

        # Timer control
        self.create_timer(self.dt, self.timer_callback)

        self.get_logger().info(
            f'Control listo | Kd={self.Kd} | Ktheta={self.Ktheta} | '
            f'umbral={self.threshold} m | esperando /set_point...'
        )
    # Callback odometry    
    def odom_callback(self, msg: Odometry):
        self.xr = msg.pose.pose.position.x
        self.yr = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.thetar = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
   
    # Callback set_point    
    def set_point_callback(self, msg: Vector3):
        if self.x_goal != msg.x or self.y_goal != msg.y:
            self.x_goal       = msg.x
            self.y_goal       = msg.y
            self.goal_reached = False
            self.get_logger().info(
                f'Nuevo goal: ({self.x_goal:.2f}, {self.y_goal:.2f})'
            )

    # Timer callback    
    def timer_callback(self):    

        # Esperar primer set_point
        if self.x_goal is None:
            return
        
       # Si ya llegamos , no hacemos nada
        if self.goal_reached:
            self._stop_robot()
            reached_msg = Bool()
            reached_msg.data = True
            self.goal_reached_pub.publish(reached_msg)
            return

        # Calcular error de pose 
        ex = self.x_goal - self.xr
        ey = self.y_goal - self.yr  

        # Error de distancia 
        ed = math.sqrt(ex ** 2 + ey ** 2)       

        # Verificar si ya llegamos
        if ed < self.threshold:
            self.goal_reached = True
            self._stop_robot()
            reached_msg = Bool()
            reached_msg.data = True
            self.goal_reached_pub.publish(reached_msg)
            self.get_logger().info(
                f'¡Meta alcanzada! Posición final: '
                f'x={self.xr:.3f} m, y={self.yr:.3f} m | '
                f'Error restante: {ed:.4f} m'
            )
            return
        # Error de angulo
        angle_to_goal = math.atan2(ey, ex)
        etheta = wrap_to_pi(angle_to_goal - self.thetar)
        
        # Control proporcional
        v = self.Kd * ed
        w = self.Ktheta * etheta

        # Saturation
        V = max(-self.v_max, min(self.v_max, v))
        w = max(-self.w_max, min(self.w_max, w))

        # Publicar comandos 
        cmd = Twist()
        cmd.linear.x = V
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)
        # Log cada segundo para no saturar 

        self._log_count += 1
        if self._log_count % 20 == 0:
            self.get_logger().info(
                f'pos=({self.xr:.2f},{self.yr:.2f}) | '
                f'goal=({self.x_goal:.2f},{self.y_goal:.2f}) | '
                f'ed={ed:.3f}m | eθ={math.degrees(etheta):.1f}° | '
                f'V={V:.3f} w={w:.3f}'
            )

    def _stop_robot(self):
        cmd = Twist()
        cmd.linear.x  = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = Control()
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
        


        


        