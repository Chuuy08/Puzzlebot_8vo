#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import (Pose, PoseArray,
                                PoseWithCovarianceStamped, TransformStamped)
from tf2_ros import TransformBroadcaster
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from scipy.ndimage import distance_transform_edt


class MCLNode(Node):

    def __init__(self):
        super().__init__('mcl_node')

        self.declare_parameter('num_particles',    500)
        self.declare_parameter('alpha1',           0.2)   # rot  noise ← rot
        self.declare_parameter('alpha2',           0.2)   # rot  noise ← trans
        self.declare_parameter('alpha3',           0.1)   # trans noise ← trans
        self.declare_parameter('alpha4',           0.1)   # trans noise ← rot
        self.declare_parameter('sigma_hit',        0.2)
        self.declare_parameter('z_hit',            0.8)
        self.declare_parameter('z_rand',           0.2)
        self.declare_parameter('laser_max_range',  6.0)
        self.declare_parameter('laser_min_range',  0.15)
        self.declare_parameter('beam_step',        10)
        self.declare_parameter('update_min_d',     0.10)
        self.declare_parameter('update_min_a',     0.10)
        self.declare_parameter('resample_interval', 2)
        self.declare_parameter('initial_pose_x',   0.0)
        self.declare_parameter('initial_pose_y',   0.0)
        self.declare_parameter('initial_pose_a',   0.0)
        self.declare_parameter('set_initial_pose', False)

        self.N          = self.get_parameter('num_particles').value
        self.alpha1     = self.get_parameter('alpha1').value
        self.alpha2     = self.get_parameter('alpha2').value
        self.alpha3     = self.get_parameter('alpha3').value
        self.alpha4     = self.get_parameter('alpha4').value
        self.sigma_hit  = self.get_parameter('sigma_hit').value
        self.z_hit      = self.get_parameter('z_hit').value
        self.z_rand     = self.get_parameter('z_rand').value
        self.laser_max  = self.get_parameter('laser_max_range').value
        self.laser_min  = self.get_parameter('laser_min_range').value
        self.beam_step  = self.get_parameter('beam_step').value
        self.upd_d      = self.get_parameter('update_min_d').value
        self.upd_a      = self.get_parameter('update_min_a').value
        self.rs_interval = self.get_parameter('resample_interval').value

        # Particles: shape (N, 3) = [x, y, theta]
        self.particles  = np.zeros((self.N, 3))
        self.weights    = np.full(self.N, 1.0 / self.N)

        # Map (received from map_server via /map)
        self.dist_map: np.ndarray | None = None
        self.map_origin = (0.0, 0.0)
        self.map_res    = 0.05
        self.map_w      = 0
        self.map_h      = 0
        self.map_cos    = 1.0   # cos(map_theta) precomputed
        self.map_sin    = 0.0   # sin(map_theta) precomputed

        self.prev_odom: np.ndarray | None = None
        self.accum_d    = 0.0
        self.accum_a    = 0.0
        self.scan_count = 0
        self.initialized = False

        if self.get_parameter('set_initial_pose').value:
            ix  = self.get_parameter('initial_pose_x').value
            iy  = self.get_parameter('initial_pose_y').value
            ia  = self.get_parameter('initial_pose_a').value
            self.particles[:, 0] = np.random.normal(ix, 0.30, self.N)
            self.particles[:, 1] = np.random.normal(iy, 0.30, self.N)
            self.particles[:, 2] = self._wrap_v(np.random.normal(ia, math.radians(15.0), self.N))
            self.weights[:] = 1.0 / self.N
            self.initialized = True
            self.get_logger().info(f'Initial pose from params: x={ix} y={iy} a={math.degrees(ia):.1f}°')

        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.map_sub  = self.create_subscription(OccupancyGrid, '/map', self._map_cb, map_qos)
        self.odom_sub = self.create_subscription(Odometry,                  '/odom',        self._odom_cb,      10)
        self.scan_sub = self.create_subscription(LaserScan,                 '/scan',        self._scan_cb,      10)
        self.init_sub = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self._init_pose_cb, 10)

        self.pose_pub  = self.create_publisher(PoseWithCovarianceStamped, 'mcl_pose',      10)
        self.cloud_pub = self.create_publisher(PoseArray,                  'particle_cloud', 10)
        self.tf_br     = TransformBroadcaster(self)

        # Keeps map→odom TF alive from startup so RViz always has a map frame
        self.create_timer(0.1, self._tf_heartbeat)

        self.get_logger().info(f'MCL node ready | N={self.N} particles | beam_step={self.beam_step}')

    # ── Map ───────────────────────────────────────────────────────────────

    def _map_cb(self, msg: OccupancyGrid):
        self.map_res    = msg.info.resolution
        self.map_w      = msg.info.width
        self.map_h      = msg.info.height
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        q = msg.info.origin.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.map_cos = math.cos(yaw)
        self.map_sin = math.sin(yaw)

        grid = np.array(msg.data, dtype=np.int8).reshape(self.map_h, self.map_w)

        # Likelihood-field: Euclidean distance (m) to nearest occupied cell
        occupied = (grid == 100)
        dist_px  = distance_transform_edt(~occupied)
        self.dist_map = dist_px * self.map_res

        self.get_logger().info(
            f'Map received: {self.map_w}×{self.map_h} px @ {self.map_res} m/px')

        # Global localization: scatter particles uniformly across all free cells
        if not self.initialized:
            self._global_localization(grid)

    def _global_localization(self, grid: np.ndarray):
        free = np.argwhere(grid == 0)   # shape (K, 2): [row, col]
        if len(free) == 0:
            return
        idx  = np.random.choice(len(free), self.N, replace=True)
        rows = free[idx, 0].astype(float)
        cols = free[idx, 1].astype(float)
        # Add sub-pixel jitter so particles don't land on grid centres
        rows += np.random.uniform(-0.5, 0.5, self.N)
        cols += np.random.uniform(-0.5, 0.5, self.N)
        # Pixel (col, row) → map frame (x, y) accounting for map rotation
        ox, oy = self.map_origin
        self.particles[:, 0] = ox + cols * self.map_res * self.map_cos - rows * self.map_res * self.map_sin
        self.particles[:, 1] = oy + cols * self.map_res * self.map_sin + rows * self.map_res * self.map_cos
        self.particles[:, 2] = np.random.uniform(-math.pi, math.pi, self.N)
        self.weights[:] = 1.0 / self.N
        self.initialized = True
        self.get_logger().info(f'Global localization: {self.N} particles across {len(free)} free cells')

    # ── Odometry motion model (Thrun et al. Table 5.6) ────────────────────

    def _odom_cb(self, msg: Odometry):
        x  = msg.pose.pose.position.x
        y  = msg.pose.pose.position.y
        q  = msg.pose.pose.orientation
        th = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        curr = np.array([x, y, th])

        if self.prev_odom is None:
            self.prev_odom = curr
            return

        dx  = curr[0] - self.prev_odom[0]
        dy  = curr[1] - self.prev_odom[1]
        dth = self._wrap(curr[2] - self.prev_odom[2])

        trans = math.sqrt(dx ** 2 + dy ** 2)
        if trans < 1e-5 and abs(dth) < 1e-5:
            self.prev_odom = curr
            return

        # Decompose into rot1 → translation → rot2
        rot1 = self._wrap(math.atan2(dy, dx) - self.prev_odom[2]) if trans > 1e-4 else 0.0
        rot2 = self._wrap(dth - rot1)

        # Noise standard deviations
        s_r1 = math.sqrt(self.alpha1 * rot1 ** 2 + self.alpha2 * trans ** 2)
        s_tr = math.sqrt(self.alpha3 * trans ** 2 + self.alpha4 * (rot1 ** 2 + rot2 ** 2))
        s_r2 = math.sqrt(self.alpha1 * rot2 ** 2 + self.alpha2 * trans ** 2)

        r1_n = rot1  - np.random.normal(0.0, s_r1, self.N)
        tr_n = trans - np.random.normal(0.0, s_tr, self.N)
        r2_n = rot2  - np.random.normal(0.0, s_r2, self.N)

        self.particles[:, 0] += tr_n * np.cos(self.particles[:, 2] + r1_n)
        self.particles[:, 1] += tr_n * np.sin(self.particles[:, 2] + r1_n)
        self.particles[:, 2]  = self._wrap_v(self.particles[:, 2] + r1_n + r2_n)

        self.accum_d += trans
        self.accum_a += abs(dth)
        self.prev_odom = curr

    # ── Sensor model (likelihood field) ───────────────────────────────────

    def _sensor_model(self, scan: LaserScan):
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        idx    = np.arange(0, len(ranges), self.beam_step)
        r_sub  = ranges[idx]
        valid  = np.isfinite(r_sub) & (r_sub >= self.laser_min) & (r_sub < self.laser_max)
        r_sub  = r_sub[valid]
        angles = (scan.angle_min + idx[valid] * scan.angle_increment)

        if len(r_sub) == 0:
            return

        norm = 1.0 / (math.sqrt(2.0 * math.pi) * self.sigma_hit)
        sig2 = self.sigma_hit ** 2
        log_w = np.zeros(self.N)

        for r, a in zip(r_sub, angles):
            # Global angle of this beam for each particle
            beam_angle = self.particles[:, 2] + a          # shape (N,)
            hx = self.particles[:, 0] + r * np.cos(beam_angle)
            hy = self.particles[:, 1] + r * np.sin(beam_angle)

            # World → map pixel (accounts for map origin rotation)
            dx = hx - self.map_origin[0]
            dy = hy - self.map_origin[1]
            col = ( dx * self.map_cos + dy * self.map_sin) / self.map_res
            row = (-dx * self.map_sin + dy * self.map_cos) / self.map_res
            col = col.astype(int)
            row = row.astype(int)

            in_bounds = (col >= 0) & (col < self.map_w) & (row >= 0) & (row < self.map_h)
            col_c = np.clip(col, 0, self.map_w - 1)
            row_c = np.clip(row, 0, self.map_h - 1)

            # Distance to nearest obstacle (infinity for out-of-bounds)
            d = np.where(in_bounds, self.dist_map[row_c, col_c], self.laser_max)

            p = (self.z_hit  * norm * np.exp(-0.5 * d ** 2 / sig2)
                 + self.z_rand / self.laser_max)

            log_w += np.log(np.maximum(p, 1e-300))

        log_w -= log_w.max()
        self.weights = np.exp(log_w)
        self.weights /= self.weights.sum()

    # ── Systematic (low-variance) resampling ─────────────────────────────
    # Keeps high-weight particles; low-weight ones are statistically eliminated.
    # One random draw → N evenly-spaced sample points → far less variance than
    # the wheel algorithm.

    def _resample(self):
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0
        step = 1.0 / self.N
        positions = (np.random.random() * step) + step * np.arange(self.N)
        indices = np.searchsorted(cumsum, positions)
        self.particles = self.particles[indices].copy()
        self.weights[:] = 1.0 / self.N

    # ── Initial pose from RViz ────────────────────────────────────────────

    def _init_pose_cb(self, msg: PoseWithCovarianceStamped):
        x  = msg.pose.pose.position.x
        y  = msg.pose.pose.position.y
        # Spread 1.5 m around the clicked point, random orientations —
        # user doesn't need to click exactly nor drag the right angle.
        self.particles[:, 0] = np.random.normal(x, 1.5, self.N)
        self.particles[:, 1] = np.random.normal(y, 1.5, self.N)
        self.particles[:, 2] = np.random.uniform(-math.pi, math.pi, self.N)
        self.weights[:] = 1.0 / self.N
        self.initialized = True
        self.get_logger().info(f'2D Pose Estimate: centro ({x:.2f}, {y:.2f}), radio 1.5 m')

    # ── TF heartbeat — keeps map frame alive from startup ─────────────────

    def _tf_heartbeat(self):
        self._publish_tf(self.get_clock().now().to_msg())

    # ── Scan callback: main MCL loop ──────────────────────────────────────

    def _scan_cb(self, scan: LaserScan):
        if self.prev_odom is None:
            return

        if self.dist_map is not None and self.initialized and \
                (self.accum_d >= self.upd_d or self.accum_a >= self.upd_a):
            self._sensor_model(scan)
            self.scan_count += 1
            if self.scan_count % self.rs_interval == 0:
                n_eff = 1.0 / float(np.sum(self.weights ** 2))
                if n_eff < self.N / 2.0:
                    self._resample()
            self.accum_d = 0.0
            self.accum_a = 0.0

        self._publish(scan.header.stamp)

    # ── Estimate & publishing ─────────────────────────────────────────────

    def _best_estimate(self):
        wx  = float(np.dot(self.weights, self.particles[:, 0]))
        wy  = float(np.dot(self.weights, self.particles[:, 1]))
        wth = math.atan2(float(np.dot(self.weights, np.sin(self.particles[:, 2]))),
                         float(np.dot(self.weights, np.cos(self.particles[:, 2]))))
        return wx, wy, wth

    def _publish_tf(self, stamp):
        wx, wy, wth = self._best_estimate()
        ox, oy, oth = self.prev_odom if self.prev_odom is not None else (0.0, 0.0, 0.0)
        dth = self._wrap(wth - oth)
        tf = TransformStamped()
        tf.header.stamp    = stamp
        tf.header.frame_id = 'map'
        tf.child_frame_id  = 'odom'
        tf.transform.translation.x = wx - (ox * math.cos(dth) - oy * math.sin(dth))
        tf.transform.translation.y = wy - (ox * math.sin(dth) + oy * math.cos(dth))
        tf.transform.rotation.z    = math.sin(dth / 2.0)
        tf.transform.rotation.w    = math.cos(dth / 2.0)
        self.tf_br.sendTransform(tf)

    def _publish(self, stamp):
        wx, wy, wth = self._best_estimate()
        qz = math.sin(wth / 2.0)
        qw = math.cos(wth / 2.0)

        pm = PoseWithCovarianceStamped()
        pm.header.stamp    = stamp
        pm.header.frame_id = 'map'
        pm.pose.pose.position.x    = wx
        pm.pose.pose.position.y    = wy
        pm.pose.pose.orientation.z = qz
        pm.pose.pose.orientation.w = qw
        dx = self.particles[:, 0] - wx
        dy = self.particles[:, 1] - wy
        pm.pose.covariance[0]  = float(np.dot(self.weights, dx * dx))
        pm.pose.covariance[1]  = float(np.dot(self.weights, dx * dy))
        pm.pose.covariance[6]  = pm.pose.covariance[1]
        pm.pose.covariance[7]  = float(np.dot(self.weights, dy * dy))
        pm.pose.covariance[35] = 0.1
        self.pose_pub.publish(pm)

        pa = PoseArray()
        pa.header.stamp    = stamp
        pa.header.frame_id = 'map'
        for p in self.particles:
            pose = Pose()
            pose.position.x    = float(p[0])
            pose.position.y    = float(p[1])
            pose.orientation.z = float(math.sin(p[2] / 2.0))
            pose.orientation.w = float(math.cos(p[2] / 2.0))
            pa.poses.append(pose)
        self.cloud_pub.publish(pa)

        self._publish_tf(stamp)

    # ── Utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def _wrap(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    @staticmethod
    def _wrap_v(a: np.ndarray) -> np.ndarray:
        return np.arctan2(np.sin(a), np.cos(a))


def main(args=None):
    rclpy.init(args=args)
    node = MCLNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
