import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion
import math

class MapPublisher(Node):
    def __init__(self):
        super().__init__('map_publisher')
        self.publisher = self.create_publisher(Marker, '/map_marker', 10)
        self.timer = self.create_timer(1.0, self.publish_marker)

    def euler_to_quaternion(self, roll, pitch, yaw):
        cr = math.cos(roll / 2)
        sr = math.sin(roll / 2)
        cp = math.cos(pitch / 2)
        sp = math.sin(pitch / 2)
        cy = math.cos(yaw / 2)
        sy = math.sin(yaw / 2)
        q = Quaternion()
        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy
        return q

    def publish_marker(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.mesh_resource = 'file:///home/jesus/robotics_ws/src/puzzlebot_description/models/map/meshes/track.stl'

        # Misma pose que en world.sdf
        marker.pose.position.x = -0.79
        marker.pose.position.y = 4.59
        marker.pose.position.z = 3.17
        marker.pose.orientation = self.euler_to_quaternion(-2.48, -0.12, 0.76)

        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 1.0
        self.publisher.publish(marker)

def main():
    rclpy.init()
    node = MapPublisher()
    rclpy.spin(node)

if __name__ == '__main__':
    main()