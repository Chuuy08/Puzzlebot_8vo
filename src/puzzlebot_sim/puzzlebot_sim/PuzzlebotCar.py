import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker
import transforms3d
import numpy as np

class PuzzlebotPublisher(Node):
    def __init__(self):
        super().__init__('puzzlebot_publisher')

        # Puzzlebot Initial Pose
        self.initial_pos_x = 1.0
        self.initial_pos_y = 1.0
        self.initial_pos_z = 0.0
        self.initial_pos_yaw = 0.0 # Checar si se ocupa modificar

        # Puzzlebot Physical Parameters
        self.R = 0.05  # Wheel radius in meters
        self.L = 0.095 * 2   # Distance between wheels in meters

        # Motion Parameters
        self.v = 0.2  # m/s
        self.w = 0.5  # rad/s

        # Wheel kinematics
        self.v_right = self.v + self.w * (self.L / 2)
        self.v_left = self.v - self.w * (self.L / 2)
        self.omega_right = self.v_right / self.R  # Angular velocity of the right wheel
        self.omega_left = self.v_left / self.R  # Angular velocity of the left wheel

        # Acumulated wheel angles
        self.theta_right = 0.0
        self.theta_left = 0.0

        self.last_time = None

        # Define Transformations
        self.define_TF()
        # Define Markers
        self.define_markers()

        # Create Transform Broadcaster
        self.tf_br = TransformBroadcaster(self)
        self.tf_br_base_footprint = TransformBroadcaster(self)
        self.tf_br_wheel_r = TransformBroadcaster(self)
        self.tf_br_wheel_l = TransformBroadcaster(self)
        self.tf_br_caster = TransformBroadcaster(self)

        # Create Marker Publisher
        self.chasis_pub = self.create_publisher(Marker, '/chasis_marker', 10)
        self.wheel_r_pub = self.create_publisher(Marker, '/wheel_r_marker', 10)
        self.wheel_l_pub = self.create_publisher(Marker, '/wheel_l_marker', 10)
        self.caster_pub = self.create_publisher(Marker, '/caster_marker', 10)

        # Create a Timer
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    # Timer Callback
    def timer_callback(self):
        time = self.get_clock().now().nanoseconds / 1e9

        # Compute delta time
        if self.last_time is None:
            self.last_time = time
            return
        dt =  time - self.last_time
        self.last_time = time

        # Integrate robot pose (circular motion)
        self.initial_pos_x +=self.v * np.cos(self.initial_pos_yaw) * dt
        self.initial_pos_y +=self.v * np.sin(self.initial_pos_yaw) * dt
        self.initial_pos_yaw += self.w * dt

        # Integrate wheel angles
        self.theta_right += self.omega_right * dt
        self.theta_left += self.omega_left * dt

        self.chasis_marker.header.stamp = self.get_clock().now().to_msg()
        self.wheel_r_marker.header.stamp = self.get_clock().now().to_msg()
        self.wheel_l_marker.header.stamp = self.get_clock().now().to_msg()
        self.caster_marker.header.stamp = self.get_clock().now().to_msg()

        # base_footprint_tf -> base_link
        self.base_link_joint_tf.header.stamp = self.get_clock().now().to_msg()
        self.base_link_joint_tf.transform.translation.x = 0.0
        self.base_link_joint_tf.transform.translation.y = 0.0
        self.base_link_joint_tf.transform.translation.z = 0.05

        # odom -> base_footprint
        self.base_footprint_tf.header.stamp = self.get_clock().now().to_msg()
        self.base_footprint_tf.transform.translation.x = self.initial_pos_x
        self.base_footprint_tf.transform.translation.y = self.initial_pos_y
        self.base_footprint_tf.transform.translation.z = self.initial_pos_z
        q_foot = transforms3d.euler.euler2quat(0, 0, self.initial_pos_yaw)
        self.base_footprint_tf.transform.rotation.x = q_foot[1]
        self.base_footprint_tf.transform.rotation.y = q_foot[2]
        self.base_footprint_tf.transform.rotation.z = q_foot[3]
        self.base_footprint_tf.transform.rotation.w = q_foot[0]

        # base_link -> wheel_r_link
        self.wheel_r_joint_tf.header.stamp = self.get_clock().now().to_msg()
        q_wheel_r = transforms3d.euler.euler2quat(0, self.theta_right, 0)
        self.wheel_r_joint_tf.transform.rotation.x = q_wheel_r[1]
        self.wheel_r_joint_tf.transform.rotation.y = q_wheel_r[2]
        self.wheel_r_joint_tf.transform.rotation.z = q_wheel_r[3]
        self.wheel_r_joint_tf.transform.rotation.w = q_wheel_r[0]

        # base_link -> wheel_l_link
        self.wheel_l_joint_tf.header.stamp = self.get_clock().now().to_msg()
        q_wheel_l = transforms3d.euler.euler2quat(0, self.theta_left, 0)
        self.wheel_l_joint_tf.transform.rotation.x = q_wheel_l[1]
        self.wheel_l_joint_tf.transform.rotation.y = q_wheel_l[2]
        self.wheel_l_joint_tf.transform.rotation.z = q_wheel_l[3]
        self.wheel_l_joint_tf.transform.rotation.w = q_wheel_l[0]

        # base_link -> caster_link
        self.caster_joint_tf.header.stamp = self.get_clock().now().to_msg()

        # send all transforms
        self.tf_br.sendTransform(self.base_link_joint_tf)
        self.tf_br_base_footprint.sendTransform(self.base_footprint_tf)
        self.tf_br_wheel_r.sendTransform(self.wheel_r_joint_tf)
        self.tf_br_wheel_l.sendTransform(self.wheel_l_joint_tf)
        self.tf_br_caster.sendTransform(self.caster_joint_tf)

        # Update Markers
        self.chasis_pub.publish(self.chasis_marker)
        self.wheel_r_pub.publish(self.wheel_r_marker)
        self.wheel_l_pub.publish(self.wheel_l_marker)
        self.caster_pub.publish(self.caster_marker)
    # Define Markers
    def define_markers(self):
        # Initialize Marker

        # Chasis Marker
        self.chasis_marker = Marker()
        self.chasis_marker.header.frame_id = 'base_link'
        self.chasis_marker.header.stamp = self.get_clock().now().to_msg()
        self.chasis_marker.id = 0
        self.chasis_marker.type = Marker.MESH_RESOURCE
        self.chasis_marker.mesh_resource = "package://puzzlebot_sim/meshes/MCR2_puzzlebot_jetson_lidar_base.stl"
        self.chasis_marker.action = Marker.ADD
        # Pose offset to align STL origin with link origin
        self.chasis_marker.pose.position.x = 0.060898
        self.chasis_marker.pose.position.y = 0.0
        self.chasis_marker.pose.position.z = 0.1
        q_chasis = transforms3d.euler.euler2quat(1.5708, 0,3.1416)  # Rotate 90 degrees around X and Z axes
        self.chasis_marker.pose.orientation.x = q_chasis[1]
        self.chasis_marker.pose.orientation.y = q_chasis[2]
        self.chasis_marker.pose.orientation.z = q_chasis[3]
        self.chasis_marker.pose.orientation.w = q_chasis[0]

        self.chasis_marker.scale.x = 1.0
        self.chasis_marker.scale.y = 1.0
        self.chasis_marker.scale.z = 1.0
        self.chasis_marker.color.r = 1.0
        self.chasis_marker.color.g = 1.0
        self.chasis_marker.color.b = 0.0
        self.chasis_marker.color.a = 1.0

        # Right Wheel Marker
        self.wheel_r_marker = Marker()
        self.wheel_r_marker.header.frame_id = 'wheel_r_link'
        self.wheel_r_marker.header.stamp = self.get_clock().now().to_msg()
        self.wheel_r_marker.id = 0
        self.wheel_r_marker.type = Marker.MESH_RESOURCE
        self.wheel_r_marker.mesh_resource = "package://puzzlebot_sim/meshes/wheel.stl"
        self.wheel_r_marker.action = Marker.ADD
        self.wheel_r_marker.pose.position.x = 0.0
        self.wheel_r_marker.pose.position.y = 0.0
        self.wheel_r_marker.pose.position.z = 0.0
        q_wheel_r = transforms3d.euler.euler2quat(0, 0, 0)
        self.wheel_r_marker.pose.orientation.x = q_wheel_r[1]
        self.wheel_r_marker.pose.orientation.y = q_wheel_r[2]
        self.wheel_r_marker.pose.orientation.z = q_wheel_r[3]
        self.wheel_r_marker.pose.orientation.w = q_wheel_r[0]
        self.wheel_r_marker.scale.x = 1.0
        self.wheel_r_marker.scale.y = 1.0
        self.wheel_r_marker.scale.z = 1.0
        self.wheel_r_marker.color.r = 0.2
        self.wheel_r_marker.color.g = 0.2
        self.wheel_r_marker.color.b = 0.2
        self.wheel_r_marker.color.a = 1.0

        # Left Wheel Marker
        self.wheel_l_marker = Marker()
        self.wheel_l_marker.header.frame_id = 'wheel_l_link'
        self.wheel_l_marker.header.stamp = self.get_clock().now().to_msg()
        self.wheel_l_marker.id = 0
        self.wheel_l_marker.type = Marker.MESH_RESOURCE
        self.wheel_l_marker.mesh_resource = "package://puzzlebot_sim/meshes/wheel.stl"
        self.wheel_l_marker.action = Marker.ADD
        self.wheel_l_marker.pose.position.x = 0.0
        self.wheel_l_marker.pose.position.y = 0.0
        self.wheel_l_marker.pose.position.z = 0.0
        q_wheel_l = transforms3d.euler.euler2quat(np.pi, 0, 0)
        self.wheel_l_marker.pose.orientation.x = q_wheel_l[1]
        self.wheel_l_marker.pose.orientation.y = q_wheel_l[2]
        self.wheel_l_marker.pose.orientation.z = q_wheel_l[3]
        self.wheel_l_marker.pose.orientation.w = q_wheel_l[0]
        self.wheel_l_marker.scale.x = 1.0
        self.wheel_l_marker.scale.y = 1.0
        self.wheel_l_marker.scale.z = 1.0
        self.wheel_l_marker.color.r = 0.2
        self.wheel_l_marker.color.g = 0.2
        self.wheel_l_marker.color.b = 0.2
        self.wheel_l_marker.color.a = 1.0

        # Caster Marker
        self.caster_marker = Marker()
        self.caster_marker.header.frame_id = 'caster_link'
        self.caster_marker.header.stamp = self.get_clock().now().to_msg()
        self.caster_marker.id = 0
        self.caster_marker.type = Marker.MESH_RESOURCE
        self.caster_marker.mesh_resource = "package://puzzlebot_sim/meshes/MCR2_caster_wheel.stl"
        self.caster_marker.action = Marker.ADD
        self.caster_marker.pose.position.x = 0.0
        self.caster_marker.pose.position.y = 0.0
        self.caster_marker.pose.position.z = 0.0
        self.caster_marker.pose.orientation.w = 1.0
        self.caster_marker.scale.x = 1.0
        self.caster_marker.scale.y = 1.0
        self.caster_marker.scale.z = 1.0
        self.caster_marker.color.r = 0.5
        self.caster_marker.color.g = 0.5
        self.caster_marker.color.b = 0.5
        self.caster_marker.color.a = 1.0
    
    # Define Transformations
    def define_TF(self):
        # Create Transform Messages
        self.base_footprint_tf = TransformStamped()
        self.base_footprint_tf.header.frame_id = 'odom'
        self.base_footprint_tf.child_frame_id = 'base_footprint'
        self.base_footprint_tf.transform.translation.x = self.initial_pos_x
        self.base_footprint_tf.transform.translation.y = self.initial_pos_y
        self.base_footprint_tf.transform.translation.z = self.initial_pos_z
        q_foot = transforms3d.euler.euler2quat(0, 0, self.initial_pos_yaw)
        self.base_footprint_tf.transform.rotation.x = q_foot[1]
        self.base_footprint_tf.transform.rotation.y = q_foot[2]
        self.base_footprint_tf.transform.rotation.z = q_foot[3]
        self.base_footprint_tf.transform.rotation.w = q_foot[0]

        # Create Transform Messages
        self.base_link_joint_tf = TransformStamped()
        self.base_link_joint_tf.header.frame_id = 'base_footprint'
        self.base_link_joint_tf.child_frame_id = 'base_link'
        self.base_link_joint_tf.transform.translation.x = 0.0
        self.base_link_joint_tf.transform.translation.y = 0.0
        self.base_link_joint_tf.transform.translation.z = 0.05
        q = transforms3d.euler.euler2quat(0, 0, 0)
        self.base_link_joint_tf.transform.rotation.x = q[1]
        self.base_link_joint_tf.transform.rotation.y = q[2]
        self.base_link_joint_tf.transform.rotation.z = q[3]
        self.base_link_joint_tf.transform.rotation.w = q[0]

        # Create Transform Messages
        self.wheel_r_joint_tf = TransformStamped()
        self.wheel_r_joint_tf.header.frame_id = 'base_link'
        self.wheel_r_joint_tf.child_frame_id = 'wheel_r_link'
        self.wheel_r_joint_tf.transform.translation.x = 0.052
        self.wheel_r_joint_tf.transform.translation.y = -0.095
        self.wheel_r_joint_tf.transform.translation.z = -0.0025
        q = transforms3d.euler.euler2quat(0, 0, 0)
        self.wheel_r_joint_tf.transform.rotation.x = q[1]
        self.wheel_r_joint_tf.transform.rotation.y = q[2]
        self.wheel_r_joint_tf.transform.rotation.z = q[3]
        self.wheel_r_joint_tf.transform.rotation.w = q[0]

        self.wheel_l_joint_tf = TransformStamped()
        self.wheel_l_joint_tf.header.frame_id = 'base_link'
        self.wheel_l_joint_tf.child_frame_id = 'wheel_l_link'
        self.wheel_l_joint_tf.transform.translation.x = 0.052
        self.wheel_l_joint_tf.transform.translation.y = 0.095
        self.wheel_l_joint_tf.transform.translation.z = -0.0025
        q = transforms3d.euler.euler2quat(0, 0, 0)
        self.wheel_l_joint_tf.transform.rotation.x = q[1]
        self.wheel_l_joint_tf.transform.rotation.y = q[2]
        self.wheel_l_joint_tf.transform.rotation.z = q[3]
        self.wheel_l_joint_tf.transform.rotation.w = q[0]

        self.caster_joint_tf = TransformStamped()
        self.caster_joint_tf.header.frame_id = 'base_link'
        self.caster_joint_tf.child_frame_id = 'caster_link'
        self.caster_joint_tf.transform.translation.x = -0.095
        self.caster_joint_tf.transform.translation.y = 0.0
        self.caster_joint_tf.transform.translation.z = -0.03
        q = transforms3d.euler.euler2quat(0, 0, 0)
        self.caster_joint_tf.transform.rotation.x = q[1]
        self.caster_joint_tf.transform.rotation.y = q[2]
        self.caster_joint_tf.transform.rotation.z = q[3]
        self.caster_joint_tf.transform.rotation.w = q[0]

def main(args=None):
    rclpy.init(args=args)
    node = PuzzlebotPublisher()
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
