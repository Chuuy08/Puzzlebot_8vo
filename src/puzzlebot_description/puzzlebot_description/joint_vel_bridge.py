import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32


class JointVelBridge(Node):

    def __init__(self):
        super().__init__('joint_vel_bridge')

        self.declare_parameter('prefix', '')
        prefix = self.get_parameter('prefix').get_parameter_value().string_value

        self.right_joint = f'{prefix}wheel_right_joint'
        self.left_joint  = f'{prefix}wheel_left_joint'

        self.pub_wr = self.create_publisher(Float32, 'wr', 10)
        self.pub_wl = self.create_publisher(Float32, 'wl', 10)

        self.create_subscription(JointState, 'joint_states', self.callback, 10)

        self.get_logger().info(
            f'JointVelBridge listo | right={self.right_joint} | left={self.left_joint}')

    def callback(self, msg: JointState):
        if self.right_joint not in msg.name or self.left_joint not in msg.name:
            return

        idx_r = msg.name.index(self.right_joint)
        idx_l = msg.name.index(self.left_joint)

        wr = Float32()
        wl = Float32()
        wr.data = float(msg.velocity[idx_r])
        wl.data = float(msg.velocity[idx_l])

        self.pub_wr.publish(wr)
        self.pub_wl.publish(wl)


def main(args=None):
    rclpy.init(args=args)
    node = JointVelBridge()
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
