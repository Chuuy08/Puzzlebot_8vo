import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState


class JointStatePublisher(Node):

    def __init__(self):
        super().__init__('joint_state_publisher')

        self.declare_parameter('sampling_time', 0.05)
        self.dt = self.get_parameter('sampling_time').value

        self.theta_r = 0.0
        self.theta_l = 0.0
        self.wr = 0.0
        self.wl = 0.0

        # Tópicos relativos — el namespace los prefija automáticamente
        self.create_subscription(Float32, 'wr', self.wr_callback, 10)
        self.create_subscription(Float32, 'wl', self.wl_callback, 10)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        self.create_timer(self.dt, self.timer_callback)
        self.get_logger().info('Joint State Publisher listo')

    def wr_callback(self, msg: Float32):
        self.wr = msg.data

    def wl_callback(self, msg: Float32):
        self.wl = msg.data

    def timer_callback(self):
        self.theta_r += self.wr * self.dt
        self.theta_l += self.wl * self.dt

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        # Sin prefijo — frame_prefix en robot_state_publisher ya lo maneja
        #js.name     = ['wheel_r_joint', 'wheel_l_joint']
        js.name     = ['right_wheel_joint', 'left_wheel_joint']
        js.position = [self.theta_r, self.theta_l]
        js.velocity = [self.wr, self.wl]
        self.joint_pub.publish(js)


def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()
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