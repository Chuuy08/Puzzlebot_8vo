#!/usr/bin/env python3
"""Republish /scan_raw with current ROS clock timestamp → /scan.

The RPLidar driver uses an internal hardware clock that drifts from the system
clock. This relay overwrites the header stamp so TF lookups work correctly.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan

# Subscribe with BEST_EFFORT to match the sllidar driver's publisher QoS.
SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)

# Publish with RELIABLE so RViz2 and Cartographer can subscribe without extra config.
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
        self.get_logger().info('Scan relay ready: /scan_raw → /scan (restamped, RELIABLE)')

    def _cb(self, msg: LaserScan):
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ScanRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
