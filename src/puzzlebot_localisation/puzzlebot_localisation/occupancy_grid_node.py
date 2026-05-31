#!/usr/bin/env python3
"""
ICP Occupancy Grid Node — builds a 2-D map from ICP pose + LaserScan.

Package : puzzlebot_localisation
File    : puzzlebot_localisation/occupancy_grid_node.py
Run     : ros2 run puzzlebot_localisation icp_map_node

Subscribes:
  ~/icp_odom  (nav_msgs/Odometry)   — robot pose from icp_node
  ~/scan      (sensor_msgs/LaserScan)
Publishes:
  ~/icp_map   (nav_msgs/OccupancyGrid)
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from .utils import quat_to_yaw


class OccupancyGridNode(Node):

    def __init__(self):
        super().__init__('icp_map_node')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('resolution',   0.05)   # m/cell
        self.declare_parameter('grid_size',    200)    # cells → 200*0.05 = 10 m
        self.declare_parameter('map_frame',    'map')
        self.declare_parameter('publish_rate', 2.0)    # Hz

        self.res        = self.get_parameter('resolution').value
        self.size       = self.get_parameter('grid_size').value
        self.map_frame  = self.get_parameter('map_frame').value
        pub_rate        = self.get_parameter('publish_rate').value

        # Grid origin: robot starts at the centre
        self.ox = -(self.size * self.res / 2.0)   # world x at col=0
        self.oy = -(self.size * self.res / 2.0)   # world y at row=0

        # ── Log-odds grid ─────────────────────────────────────────────
        # 0 = unknown (never visited).
        # Positive → probably occupied; Negative → probably free.
        self.log_odds = np.zeros((self.size, self.size), dtype=np.float32)

        # Asymmetric update: slow to erase, fast to mark occupied.
        # This makes the map robust to small pose errors from ICP drift.
        # A wall needs many contradicting rays to be erased (unlikely if drift is small).
        self.L_FREE = -0.10   # very slow to erase walls (noise resistant)
        self.L_OCC  =  0.7   # slightly softer occupied update (reduces noise spikes)
        self.L_MIN  = -2.0
        self.L_MAX  =  4.0   # lower saturation → walls can be corrected if pose improves

        # ── Robot pose (updated from icp_odom) ────────────────────────
        self.rx = 0.0
        self.ry = 0.0
        self.rth = 0.0

        # ── ROS I/O ───────────────────────────────────────────────────
        self.odom_sub = self.create_subscription(
            Odometry, 'icp_odom', self._odom_cb, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self._scan_cb, 10)
        self.map_pub = self.create_publisher(OccupancyGrid, 'icp_map', 10)
        self.create_timer(1.0 / pub_rate, self._publish_map)

        self.get_logger().info(
            f'ICP map node ready | {self.size}×{self.size} cells '
            f'@ {self.res} m/cell = {self.size*self.res:.1f} m side')

    # ══════════════════════════════════════════════════════════════════
    # Callbacks
    # ══════════════════════════════════════════════════════════════════

    def _odom_cb(self, msg: Odometry):
        """Store latest robot pose from ICP node."""
        self.rx = msg.pose.pose.position.x
        self.ry = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.rth = quat_to_yaw(q)

    def _scan_cb(self, msg: LaserScan):
        """Update occupancy grid with one laser scan."""

        # 1. Robot cell
        r0, c0 = self._world_to_cell(self.rx, self.ry)
        if not self._in_bounds(r0, c0):
            self.get_logger().warn('Robot outside grid — increase grid_size', once=True)
            return

        n = len(msg.ranges)
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        local_angles = msg.angle_min + np.arange(n) * msg.angle_increment
        # Beam angles in world frame (add robot heading)
        world_angles = local_angles + self.rth

        for i in range(n):
            r = ranges[i]
            if not (msg.range_min <= r <= msg.range_max) or not math.isfinite(r):
                continue

            # 2. Hit point in world frame
            hx = self.rx + r * math.cos(world_angles[i])
            hy = self.ry + r * math.sin(world_angles[i])
            r1, c1 = self._world_to_cell(hx, hy)

            # 3. Ray trace from robot to hit point
            #    All cells along the ray (vectorised Bresenham via linspace)
            rows, cols = self._raytrace(r0, c0, r1, c1)
            if len(rows) == 0:
                continue

            # 4. Mark ray cells as FREE (all but the last)
            #    Mark hit cell as OCCUPIED
            if len(rows) > 1:
                self.log_odds[rows[:-1], cols[:-1]] += self.L_FREE
            if self._in_bounds(r1, c1):
                self.log_odds[r1, c1] += self.L_OCC

        # 5. Clamp to prevent saturation (keeps map updatable)
        np.clip(self.log_odds, self.L_MIN, self.L_MAX, out=self.log_odds)

    # ══════════════════════════════════════════════════════════════════
    # Grid utilities
    # ══════════════════════════════════════════════════════════════════

    def _world_to_cell(self, wx: float, wy: float):
        """World coordinates (m) → (row, col) grid indices."""
        col = int((wx - self.ox) / self.res)
        row = int((wy - self.oy) / self.res)
        return row, col

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def _raytrace(self, r0: int, c0: int, r1: int, c1: int):
        """
        Return (rows, cols) for all grid cells along the segment (r0,c0)→(r1,c1).

        Uses np.linspace interpolation — vectorised, equivalent to Bresenham.
        Cells outside the grid are silently dropped.
        """
        n = max(abs(r1 - r0), abs(c1 - c0)) + 1
        rows = np.round(np.linspace(r0, r1, n)).astype(int)
        cols = np.round(np.linspace(c0, c1, n)).astype(int)
        valid = (rows >= 0) & (rows < self.size) & (cols >= 0) & (cols < self.size)
        return rows[valid], cols[valid]

    # ══════════════════════════════════════════════════════════════════
    # Map publishing
    # ══════════════════════════════════════════════════════════════════

    def _publish_map(self):
        """
        Convert log-odds grid to ROS OccupancyGrid and publish.

        Conversion:
          probability = 1 - 1/(1 + exp(log_odds))   ← inverse logit
          ROS value   = round(probability * 100)      ← 0=free, 100=occupied
          Unknown     = -1  (cells never visited)
        """
        prob = 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))

        data = np.full((self.size, self.size), -1, dtype=np.int8)
        visited = self.log_odds != 0.0
        data[visited] = np.clip(prob[visited] * 100, 0, 100).astype(np.int8)

        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame

        msg.info.resolution = self.res
        msg.info.width      = self.size
        msg.info.height     = self.size
        msg.info.origin.position.x  = self.ox
        msg.info.origin.position.y  = self.oy
        msg.info.origin.orientation.w = 1.0

        # ROS OccupancyGrid is row-major, row 0 = bottom of map
        msg.data = data.flatten().tolist()
        self.map_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
