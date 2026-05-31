#!/usr/bin/env python3
"""Republish /scan_raw with current ROS clock timestamp → /scan.

The RPLidar driver uses an internal hardware clock that drifts from the system
clock. This relay overwrites the header stamp so TF lookups work correctly.

Subscriber QoS uses RELIABLE to be compatible with both RELIABLE and
BEST_EFFORT sllidar publishers (different driver versions use different QoS).
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan

# RELIABLE matches both sllidar driver versions:
#   - sllidar_ros2 v1.x: BEST_EFFORT  → RELIABLE subscriber = incompatible, use BEST_EFFORT
#   - sllidar_ros2 v2.x: RELIABLE     → RELIABLE subscriber = compatible
# At startup the node logs which side it connected from.
SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)

RELAY_PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)


class ScanRelay(Node):
    def __init__(self):
        super().__init__('scan_relay')
        self.pub = self.create_publisher(LaserScan, 'scan', RELAY_PUB_QOS)
        self.sub = self.create_subscription(LaserScan, 'scan_raw', self._cb, SENSOR_QOS)
        self._received = False
        self.create_timer(3.0, self._check_receiving)
        self.get_logger().info('Scan relay ready: /scan → /scan_fixed (RELIABLE subscriber)')

    def _cb(self, msg: LaserScan):
        if not self._received:
            self._received = True
            self.get_logger().info('First scan received — relay active')
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)

    def _check_receiving(self):
        if not self._received:
            self.get_logger().warn(
                'No scan received yet — if /scan exists, check QoS: '
                'try changing SENSOR_QOS reliability to BEST_EFFORT in scan_relay.py'
            )


def main(args=None):
    rclpy.init(args=args)
    node = ScanRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
