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

    # Mixture MCL EWMA rates (Thrun §8.3.3)
    _ALPHA_SLOW = 0.001
    _ALPHA_FAST = 0.1

    def __init__(self):
        super().__init__('mcl_node')

        self.declare_parameter('num_particles',    1000)
        self.declare_parameter('alpha1',           0.2)
        self.declare_parameter('alpha2',           0.2)
        self.declare_parameter('alpha3',           0.1)
        self.declare_parameter('alpha4',           0.1)
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

        self.particles  = np.zeros((self.N, 3))
        self.weights    = np.full(self.N, 1.0 / self.N)

        # Map state
        self.dist_map: np.ndarray | None  = None
        self.free_cells: np.ndarray | None = None   # free-cell (row,col) index
        self.map_origin = (0.0, 0.0)
        self.map_res    = 0.05
        self.map_w      = 0
        self.map_h      = 0
        self.map_cos    = 1.0
        self.map_sin    = 0.0

        # Odometry / motion accumulation
        self.prev_odom: np.ndarray | None = None
        self.accum_d    = 0.0
        self.accum_a    = 0.0
        self.scan_count = 0
        self.initialized = False

        # Mixture MCL weight EWMAs: track long-term vs short-term filter quality
        self.w_slow = 0.0
        self.w_fast = 0.0

        # Best estimate cached from pre-resample distribution.
        # Using post-resample weights (uniform 1/N) would bias the pose toward
        # the map center whenever random global particles are injected.
        # (wx, wy, wth, cov_x, cov_xy, cov_y)
        self._mcl_pose: tuple | None = None

        if self.get_parameter('set_initial_pose').value:
            ix = self.get_parameter('initial_pose_x').value
            iy = self.get_parameter('initial_pose_y').value
            ia = self.get_parameter('initial_pose_a').value
            self.particles[:, 0] = np.random.normal(ix, 0.30, self.N)
            self.particles[:, 1] = np.random.normal(iy, 0.30, self.N)
            self.particles[:, 2] = self._wrap_v(np.random.normal(ia, math.radians(15.0), self.N))
            self.weights[:] = 1.0 / self.N
            self.initialized = True
            self.get_logger().info(
                f'Initial pose from params: x={ix} y={iy} a={math.degrees(ia):.1f}°')

        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.map_sub  = self.create_subscription(OccupancyGrid, '/map', self._map_cb, map_qos)
        self.odom_sub = self.create_subscription(Odometry,                  '/odom',        self._odom_cb,      10)
        self.scan_sub = self.create_subscription(LaserScan,                 '/scan',        self._scan_cb,      10)
        self.init_sub = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self._init_pose_cb, 10)

        self.pose_pub  = self.create_publisher(PoseWithCovarianceStamped, 'mcl_pose',       10)
        self.cloud_pub = self.create_publisher(PoseArray,                  'particle_cloud', 10)
        self.tf_br     = TransformBroadcaster(self)

        self.create_timer(0.1, self._tf_heartbeat)

        self.get_logger().info(
            f'MCL (Mixture) ready | N={self.N} particles | beam_step={self.beam_step}')

    # ── Map ───────────────────────────────────────────────────────────────

    def _map_cb(self, msg: OccupancyGrid):
        self.map_res    = msg.info.resolution
        self.map_w      = msg.info.width
        self.map_h      = msg.info.height
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        q   = msg.info.origin.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.map_cos = math.cos(yaw)
        self.map_sin = math.sin(yaw)

        grid = np.array(msg.data, dtype=np.int8).reshape(self.map_h, self.map_w)

        # Free cells index — used for random particle injection
        self.free_cells = np.argwhere(grid == 0)

        occupied = (grid == 100)
        self.dist_map = distance_transform_edt(~occupied) * self.map_res

        self.get_logger().info(
            f'Map received: {self.map_w}×{self.map_h} px @ {self.map_res} m/px | '
            f'{len(self.free_cells)} free cells')

        if not self.initialized:
            self._global_localization()

    # ── Global localization ───────────────────────────────────────────────

    def _global_localization(self):
        if self.free_cells is None or len(self.free_cells) == 0:
            return
        self.particles = self._sample_free_cells(self.N)
        self.weights[:] = 1.0 / self.N
        self.w_slow = 0.0
        self.w_fast = 0.0
        self.initialized = True
        self.get_logger().info(
            f'Global localization: {self.N} particles across {len(self.free_cells)} free cells')

    def _sample_free_cells(self, n: int) -> np.ndarray:
        """Return n random poses (x, y, θ) sampled from free map cells."""
        idx  = np.random.choice(len(self.free_cells), n, replace=True)
        rows = self.free_cells[idx, 0].astype(float) + np.random.uniform(-0.5, 0.5, n)
        cols = self.free_cells[idx, 1].astype(float) + np.random.uniform(-0.5, 0.5, n)
        ox, oy = self.map_origin
        x  = ox + cols * self.map_res * self.map_cos - rows * self.map_res * self.map_sin
        y  = oy + cols * self.map_res * self.map_sin + rows * self.map_res * self.map_cos
        th = np.random.uniform(-math.pi, math.pi, n)
        return np.column_stack([x, y, th])

    def _sample_near_estimate(self, wx: float, wy: float, wth: float,
                               n: int, r_xy: float) -> np.ndarray:
        """Sample n poses from a Gaussian near the best estimate, keeping only free cells.

        r_xy: 1-σ spread in metres (derived from weighted particle covariance).
        Falls back to global free-cell sampling for candidates that land in obstacles.
        """
        oversample = max(n * 4, n + 100)
        x  = np.random.normal(wx,  r_xy, oversample)
        y  = np.random.normal(wy,  r_xy, oversample)
        th = self._wrap_v(np.random.normal(wth, 0.35, oversample))   # ±20° 1σ

        # World → map pixel
        dx = x - self.map_origin[0]
        dy = y - self.map_origin[1]
        col = ( dx * self.map_cos + dy * self.map_sin) / self.map_res
        row = (-dx * self.map_sin + dy * self.map_cos) / self.map_res
        col_i = col.astype(int)
        row_i = row.astype(int)

        in_bounds = ((col_i >= 0) & (col_i < self.map_w) &
                     (row_i >= 0) & (row_i < self.map_h))
        col_c = np.clip(col_i, 0, self.map_w - 1)
        row_c = np.clip(row_i, 0, self.map_h - 1)

        # dist_map == 0 → occupied; dist_map > 0 → free
        free = in_bounds & (self.dist_map[row_c, col_c] > 0)
        x_f, y_f, th_f = x[free], y[free], th[free]

        if len(x_f) >= n:
            pick = np.random.choice(len(x_f), n, replace=False)
            return np.column_stack([x_f[pick], y_f[pick], th_f[pick]])

        # Not enough free hits (dense obstacle area): fill remainder globally
        n_extra = n - len(x_f)
        global_p = self._sample_free_cells(n_extra)
        if len(x_f) == 0:
            return global_p
        return np.vstack([np.column_stack([x_f, y_f, th_f]), global_p])

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

        rot1 = self._wrap(math.atan2(dy, dx) - self.prev_odom[2]) if trans > 1e-4 else 0.0
        rot2 = self._wrap(dth - rot1)

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
        angles = scan.angle_min + idx[valid] * scan.angle_increment

        if len(r_sub) == 0:
            return

        norm = 1.0 / (math.sqrt(2.0 * math.pi) * self.sigma_hit)
        sig2 = self.sigma_hit ** 2
        log_w = np.zeros(self.N)

        for r, a in zip(r_sub, angles):
            beam_angle = self.particles[:, 2] + a
            hx = self.particles[:, 0] + r * np.cos(beam_angle)
            hy = self.particles[:, 1] + r * np.sin(beam_angle)

            dx = hx - self.map_origin[0]
            dy = hy - self.map_origin[1]
            col = ( dx * self.map_cos + dy * self.map_sin) / self.map_res
            row = (-dx * self.map_sin + dy * self.map_cos) / self.map_res
            col = col.astype(int)
            row = row.astype(int)

            in_bounds = (col >= 0) & (col < self.map_w) & (row >= 0) & (row < self.map_h)
            col_c = np.clip(col, 0, self.map_w - 1)
            row_c = np.clip(row, 0, self.map_h - 1)
            d = np.where(in_bounds, self.dist_map[row_c, col_c], self.laser_max)

            p = (self.z_hit  * norm * np.exp(-0.5 * d ** 2 / sig2)
                 + self.z_rand / self.laser_max)

            log_w += np.log(np.maximum(p, 1e-300))

        # ── Mixture MCL quality tracking ─────────────────────────────────
        # avg_log_w / n_beams ∈ [log(z_rand/max), log(z_hit*norm)]
        n_beams    = len(r_sub)
        avg_log_pb = float(log_w.mean()) / n_beams      # per-beam mean over particles

        log_min = math.log(max(1e-300, self.z_rand / self.laser_max))
        log_hi  = math.log(max(1e-300, self.z_hit * norm))
        rng     = log_hi - log_min
        quality = (avg_log_pb - log_min) / rng if rng > 0.0 else 0.5
        quality = max(0.0, min(1.0, quality))

        self.w_slow += self._ALPHA_SLOW * (quality - self.w_slow)
        self.w_fast += self._ALPHA_FAST * (quality - self.w_fast)
        # ─────────────────────────────────────────────────────────────────

        log_w -= log_w.max()
        self.weights = np.exp(log_w)
        self.weights /= self.weights.sum()

        # Cache best estimate NOW (pre-resample, unbiased by injected particles).
        wx  = float(np.dot(self.weights, self.particles[:, 0]))
        wy  = float(np.dot(self.weights, self.particles[:, 1]))
        wth = math.atan2(float(np.dot(self.weights, np.sin(self.particles[:, 2]))),
                         float(np.dot(self.weights, np.cos(self.particles[:, 2]))))
        dx_p = self.particles[:, 0] - wx
        dy_p = self.particles[:, 1] - wy
        cov_x  = float(np.dot(self.weights, dx_p * dx_p))
        cov_xy = float(np.dot(self.weights, dx_p * dy_p))
        cov_y  = float(np.dot(self.weights, dy_p * dy_p))
        self._mcl_pose = (wx, wy, wth, cov_x, cov_xy, cov_y)

    # ── Mixture MCL resampling ────────────────────────────────────────────

    def _resample(self):
        # Adaptive recovery fraction: inject more random particles when the
        # filter quality (w_fast) drops well below its long-term average (w_slow).
        if self.w_slow > 0.05:
            p_rand = max(0.0, 1.0 - self.w_fast / self.w_slow)
        else:
            p_rand = 0.0

        # Always inject at least 5 % random particles to prevent collapse
        n_rand_min = max(1, self.N // 20)
        n_rand = max(n_rand_min, int(self.N * p_rand))
        n_rand = min(n_rand, self.N - 1)          # keep at least 1 systematic slot
        n_keep = self.N - n_rand

        if n_rand > n_rand_min:
            self.get_logger().info(
                f'Mixture MCL: injecting {n_rand}/{self.N} random particles '
                f'(q_slow={self.w_slow:.3f} q_fast={self.w_fast:.3f})')

        # Systematic (low-variance) resampling for n_keep particles
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0
        step      = 1.0 / n_keep
        positions = (np.random.random() * step) + step * np.arange(n_keep)
        indices   = np.searchsorted(cumsum, positions)
        kept      = self.particles[indices].copy()

        # ── Adaptive local/global injection split ─────────────────────────
        # confidence = w_fast ∈ [0, 1]:
        #   high → mostly local (tight Gaussian near estimate) — fast convergence
        #   low  → mostly global (full map) — broad recovery
        confidence = max(0.0, min(1.0, self.w_fast))
        n_local  = int(n_rand * confidence)
        n_global = n_rand - n_local

        parts: list[np.ndarray] = [kept]

        if n_local > 0 and self._mcl_pose is not None:
            wx, wy, wth, cov_x, _, cov_y = self._mcl_pose
            # Local radius from particle spread, clamped to sensible range
            r_xy = float(np.clip(math.sqrt(cov_x + cov_y), 0.15, 1.5))
            parts.append(self._sample_near_estimate(wx, wy, wth, n_local, r_xy))
        elif n_local > 0:
            n_global += n_local   # fallback: add to global if no cache yet

        if n_global > 0:
            if self.free_cells is not None and len(self.free_cells) > 0:
                parts.append(self._sample_free_cells(n_global))
            else:
                parts.append(kept[np.random.choice(n_keep, n_global, replace=True)])

        self.particles = np.vstack(parts)
        self.weights[:] = 1.0 / self.N

    # ── Initial pose from RViz ────────────────────────────────────────────

    def _init_pose_cb(self, msg: PoseWithCovarianceStamped):
        x  = msg.pose.pose.position.x
        y  = msg.pose.pose.position.y
        self.particles[:, 0] = np.random.normal(x, 1.5, self.N)
        self.particles[:, 1] = np.random.normal(y, 1.5, self.N)
        self.particles[:, 2] = np.random.uniform(-math.pi, math.pi, self.N)
        self.weights[:] = 1.0 / self.N
        self.w_slow = 0.0
        self.w_fast = 0.0
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
            # Mixture MCL always resamples — the random injection is what prevents collapse.
            if self.scan_count % self.rs_interval == 0:
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
        if self._mcl_pose is not None:
            wx, wy, wth = self._mcl_pose[:3]
        else:
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
        # Use the pre-resample cached estimate so injected random particles
        # don't bias the published pose toward the map centre.
        if self._mcl_pose is not None:
            wx, wy, wth, cov_x, cov_xy, cov_y = self._mcl_pose
        else:
            wx, wy, wth = self._best_estimate()
            dx_p = self.particles[:, 0] - wx
            dy_p = self.particles[:, 1] - wy
            cov_x  = float(np.dot(self.weights, dx_p * dx_p))
            cov_xy = float(np.dot(self.weights, dx_p * dy_p))
            cov_y  = float(np.dot(self.weights, dy_p * dy_p))

        qz = math.sin(wth / 2.0)
        qw = math.cos(wth / 2.0)

        pm = PoseWithCovarianceStamped()
        pm.header.stamp    = stamp
        pm.header.frame_id = 'map'
        pm.pose.pose.position.x    = wx
        pm.pose.pose.position.y    = wy
        pm.pose.pose.orientation.z = qz
        pm.pose.pose.orientation.w = qw
        pm.pose.covariance[0]  = cov_x
        pm.pose.covariance[1]  = cov_xy
        pm.pose.covariance[6]  = cov_xy
        pm.pose.covariance[7]  = cov_y
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
