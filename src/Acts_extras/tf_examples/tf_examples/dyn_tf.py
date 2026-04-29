import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import transforms3d
import numpy as np


class FramePublisher(Node):
    def __init__(self):
        super().__init__('frame_publisher')
        self.t = TransformStamped()
        self.t2 = TransformStamped()

        # Create TrabsformBroadcaster 
        self.tf_br1 = TransformBroadcaster(self)
        self.tf_br2 = TransformBroadcaster(self)

        # Create a timer
        timer_period = 0.1 # seconds
        self.timer = self.create_timer(timer_period, self.timer_cb)

        # Varables to be used
        self.start_time = self.get_clock().now()
        self.omega = 0.1

    def timer_cb(self):
            elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9


            self.t.header.stamp = self.get_clock().now().to_msg()
            self.t.header.frame_id = 'world'
            self.t.child_frame_id = 'moving_robot_3'
            self.t.transform.translation.x = 0.5 * np.sin(self.omega * elapsed_time)
            self.t.transform.translation.y = 0.5 * np.cos(self.omega * elapsed_time)
            self.t.transform.translation.z = 0.0
            q = transforms3d.euler.euler2quat(0, 0, -self.omega*elapsed_time) # Input euler2quaternion(roll, pitch, yaw), output q=(w, x, y, z)
            self.t.transform.rotation.x = q[1]
            self.t.transform.rotation.y = q[2]
            self.t.transform.rotation.z = q[3]
            self.t.transform.rotation.w = q[0]


            self.t2.header.stamp = self.get_clock().now().to_msg()
            self.t2.header.frame_id = 'world'
            self.t2.child_frame_id = 'moving_robot_4'
            self.t2.transform.translation.x = 1.0
            self.t2.transform.translation.y = 1.0
            self.t2.transform.translation.z = 1.0
            q = transforms3d.euler.euler2quat(elapsed_time,elapsed_time,0) # Input euler2quaternion(roll, pitch, yaw), output q=(w, x, y, z)
            self.t2.transform.rotation.x = q[1]
            self.t2.transform.rotation.y = q[2]
            self.t2.transform.rotation.z = q[3]
            self.t2.transform.rotation.w = q[0]

            self.tf_br1.sendTransform(self.t)
            self.tf_br2.sendTransform(self.t2)


def main(args=None):
    rclpy.init(args=args)

    node = FramePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()

if __name__ == '__main__':
    main()