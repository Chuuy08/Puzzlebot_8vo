import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
import transforms3d
import numpy as np

class PuzzleBotPublisher(Node):
    def __init__(self):
        super().__init__('puzzlebot_publisher')

        # Puzzlebot Initial Pose
        self.initial_pos_x = 1.0
        self.initial_pos_y = 1.0
        self.initial_pos_z = 0.0
        self.initial_pos_yaw = 0.0

        # Puzzlebot Physical Parameters
        self.R = 0.05  # Wheel radius in meters
        self.L = 0.095 * 2   # Distance between wheels in meters

        # Motion Parameters
        self.v = 0.2  # m/s
        self.w = 0.5  # rad/s

        # Wheel kinematics
        self.v_right = self.v + self.w * (self.L / 2)
        self.v_left = self.v - self.w * (self.L / 2)
        self.omega_right = self.v_right / self.R
        self.omega_left = self.v_left / self.R

        # Accumulated wheel angles
        self.theta_right = 0.0
        self.theta_left = 0.0

        self.last_time = None

        # Create Transform Broadcaster
        self.tf_br = TransformBroadcaster(self)

        # Create JointState Publisher
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)

        # Create a Timer
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        time = self.get_clock().now().nanoseconds / 1e9

        # Compute delta time
        if self.last_time is None:
            self.last_time = time
            return
        dt = time - self.last_time
        self.last_time = time

        # Integrate robot pose (circular motion)
        self.initial_pos_x += self.v * np.cos(self.initial_pos_yaw) * dt
        self.initial_pos_y += self.v * np.sin(self.initial_pos_yaw) * dt
        self.initial_pos_yaw += self.w * dt

        # Integrate wheel angles
        self.theta_right += self.omega_right * dt
        self.theta_left += self.omega_left * dt

        current_time_msg = self.get_clock().now().to_msg()

        # odom -> base_footprint tf
        tf = TransformStamped()
        tf.header.stamp = current_time_msg
        tf.header.frame_id = 'odom'
        #tf.child_frame_id = 'base_footprint'
        tf.child_frame_id = 'base_link'
        tf.transform.translation.x = self.initial_pos_x
        tf.transform.translation.y = self.initial_pos_y
        tf.transform.translation.z = self.initial_pos_z
        q_foot = transforms3d.euler.euler2quat(0, 0, self.initial_pos_yaw)
        tf.transform.rotation.w = q_foot[0]
        tf.transform.rotation.x = q_foot[1]
        tf.transform.rotation.y = q_foot[2]
        tf.transform.rotation.z = q_foot[3]

        # send transform
        self.tf_br.sendTransform(tf)

        # publish joint states
        joint_state = JointState()
        joint_state.header.stamp = current_time_msg
        #joint_state.name = ['wheel_l_joint', 'wheel_r_joint']
        joint_state.name = ['left_wheel_joint', 'right_wheel_joint']
        joint_state.position = [self.theta_left, self.theta_right]
        
        self.joint_state_pub.publish(joint_state)

def main(args=None):
    rclpy.init(args=args)
    node = PuzzleBotPublisher()
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