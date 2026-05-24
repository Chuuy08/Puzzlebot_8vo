#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster


class ICPNode(Node):

    def __init__(self):
        super().__init__('icp_node')

        self.declare_parameter('max_iterations',              30)
        self.declare_parameter('convergence_threshold',       1e-4)
        self.declare_parameter('max_correspondence_dist',     0.5)
        self.declare_parameter('downsample_step',             3)
        self.declare_parameter('min_points',                  20)
        self.declare_parameter('map_frame',                   'map')
        self.declare_parameter('base_frame',                  'base_footprint')
        self.declare_parameter('laser_x_offset',              0.0)
        self.declare_parameter('laser_y_offset',              0.0)
        self.declare_parameter('use_odom_init',               True)
        self.declare_parameter('max_rotation_correction_deg', 3.0)
        self.declare_parameter('skip_icp_ang_vel_thresh',     0.15)
        self.declare_parameter('loop_check_every_n_scans',    15)
        self.declare_parameter('min_scans_before_loop',       80)
        self.declare_parameter('loop_closure_score_thresh',   0.12)
        self.declare_parameter('keyframe_dist_m',             0.25)
        self.declare_parameter('keyframe_angle_deg',          15.0)

        self.max_iter          = self.get_parameter('max_iterations').value
        self.tol               = self.get_parameter('convergence_threshold').value
        self.max_dist          = self.get_parameter('max_correspondence_dist').value
        self.step              = self.get_parameter('downsample_step').value
        self.min_pts           = self.get_parameter('min_points').value
        self.map_frame         = self.get_parameter('map_frame').value
        self.base_frame        = self.get_parameter('base_frame').value
        self.lx                = self.get_parameter('laser_x_offset').value
        self.ly                = self.get_parameter('laser_y_offset').value
        self.use_odom          = self.get_parameter('use_odom_init').value
        self.max_rot_correction = math.radians(self.get_parameter('max_rotation_correction_deg').value)
        self.loop_check_period = self.get_parameter('loop_check_every_n_scans').value
        self.min_scans_loop    = self.get_parameter('min_scans_before_loop').value
        self.loop_score_thresh = self.get_parameter('loop_closure_score_thresh').value
        self.skip_icp_thresh   = self.get_parameter('skip_icp_ang_vel_thresh').value
        self.kf_dist           = self.get_parameter('keyframe_dist_m').value
        self.kf_angle          = math.radians(self.get_parameter('keyframe_angle_deg').value)

        self.pose: np.ndarray         = np.zeros(3)
        self.odom_pose: np.ndarray | None = None
        self.odom_ang_vel: float      = 0.0
        self.ref_scan: np.ndarray | None = None
        self.ref_pose: np.ndarray     = np.zeros(3)
        self.initial_scan: np.ndarray | None = None
        self.scan_count               = 0
        self.loop_closed              = False

        _I3 = np.eye(3, dtype=np.float64)
        self.odom_cov: np.ndarray    = _I3 * 1e-4
        self.ref_cov:  np.ndarray    = _I3 * 1e-4
        self.ekf_cov_2d: np.ndarray  = np.eye(2) * 1e-4  # EKF-corrected x,y covariance

        self.scan_sub = self.create_subscription(LaserScan, 'scan',  self._scan_cb, 10)
        self.odom_sub = self.create_subscription(Odometry,  '/odom', self._odom_cb, 10)
        self.pose_pub = self.create_publisher(PoseStamped, 'icp_pose', 10)
        self.odom_pub = self.create_publisher(Odometry,    'icp_odom', 10)
        self.tf_br    = TransformBroadcaster(self)

        self.get_logger().info(
            f'ICP node ready | max_iter={self.max_iter} | max_dist={self.max_dist} m | step={self.step}')

    # ── Point cloud helpers ───────────────────────────────────────────

    def _scan_to_robot_frame(self, msg: LaserScan) -> np.ndarray:
        n = len(msg.ranges)
        angles = msg.angle_min + np.arange(n) * msg.angle_increment
        r = np.asarray(msg.ranges, dtype=np.float64)
        valid = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
        r, a = r[valid], angles[valid]
        pts = np.column_stack([r * np.cos(a), r * np.sin(a)])
        pts[:, 0] += self.lx
        pts[:, 1] += self.ly
        return pts

    @staticmethod
    def _rigid2d(pts: np.ndarray, x: float, y: float, th: float) -> np.ndarray:
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        return (R @ pts.T).T + np.array([x, y])

    # ── ICP core ─────────────────────────────────────────────────────

    def _nearest_neighbors(self, src: np.ndarray, tgt: np.ndarray):
        diff = src[:, np.newaxis, :] - tgt[np.newaxis, :, :]
        d2   = (diff ** 2).sum(axis=2)
        idx  = np.argmin(d2, axis=1)
        dist = np.sqrt(d2[np.arange(len(src)), idx])
        return idx, dist

    def _icp(self, src: np.ndarray, tgt: np.ndarray):
        cur   = src.copy()
        R_acc = np.eye(2)
        t_acc = np.zeros(2)
        prev_err = np.inf

        for it in range(self.max_iter):
            idx, dist = self._nearest_neighbors(cur, tgt)
            mask = dist < self.max_dist
            if mask.sum() < self.min_pts:
                self.get_logger().warn(f'ICP iter {it}: only {mask.sum()} inliers — aborting')
                return None

            p, q = cur[mask], tgt[idx[mask]]
            mu_p, mu_q = p.mean(0), q.mean(0)
            H = (p - mu_p).T @ (q - mu_q)
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1] *= -1
                R = Vt.T @ U.T
            t = mu_q - R @ mu_p

            cur   = (R @ cur.T).T + t
            R_acc = R @ R_acc
            t_acc = R @ t_acc + t

            err = dist[mask].mean()
            if abs(prev_err - err) < self.tol:
                break
            prev_err = err

        dx, dy = t_acc
        dtheta  = math.atan2(R_acc[1, 0], R_acc[0, 0])
        return dx, dy, dtheta, prev_err

    # ── Odometry callback ─────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        x  = msg.pose.pose.position.x
        y  = msg.pose.pose.position.y
        q  = msg.pose.pose.orientation
        th = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.odom_pose    = np.array([x, y, th])
        self.odom_ang_vel = abs(msg.twist.twist.angular.z)

        c = msg.pose.covariance
        self.odom_cov = np.array([
            [c[0],  c[1],  c[5]],
            [c[6],  c[7],  c[11]],
            [c[30], c[31], c[35]],
        ], dtype=np.float64)

    # ── Scan callback ─────────────────────────────────────────────────

    def _scan_cb(self, msg: LaserScan):
        pts_robot = self._scan_to_robot_frame(msg)[::self.step]
        if len(pts_robot) < self.min_pts:
            self.get_logger().warn('Too few valid scan points — skipping')
            return

        init_pose = self.odom_pose.copy() if (self.use_odom and self.odom_pose is not None) \
                    else self.pose.copy()
        pts_map = self._rigid2d(pts_robot, *init_pose)

        if self.ref_scan is None:
            self.ref_scan     = pts_map
            self.ref_pose     = init_pose.copy()
            self.ref_cov      = self.odom_cov.copy()
            self.initial_scan = pts_map.copy()
            self.pose         = init_pose.copy()
            self.get_logger().info('ICP: reference scan initialised — tracking started')
            self._publish(msg.header.stamp)
            return

        if self.odom_ang_vel > self.skip_icp_thresh:
            self.pose = init_pose.copy()
            self.scan_count += 1
            self._publish(msg.header.stamp)
            return

        result = self._icp(pts_map, self.ref_scan)

        if result is not None:
            dx, dy, dtheta, score = result
            s = max(score, 0.02)

            # ── x,y EKF ──────────────────────────────────────────────
            P2 = (self.odom_cov - self.ref_cov)[:2, :2]
            P2 = (P2 + P2.T) / 2.0
            P2 = P2 - min(np.linalg.eigvalsh(P2).min(), 0) * np.eye(2)
            P2 += np.eye(2) * 1e-6
            R2 = np.eye(2) * (s ** 2)
            K2 = P2 @ np.linalg.inv(P2 + R2)
            corr_xy = K2 @ np.array([dx, dy])
            self.ekf_cov_2d = (np.eye(2) - K2) @ P2

            # ── θ EKF — small gain to slowly correct heading drift ───
            # R_theta is 25× larger than R_xy so the gain is much smaller.
            # This prevents sudden map rotations while still correcting
            # accumulated heading error over many scans.
            p_th = max(self.odom_cov[2, 2] - self.ref_cov[2, 2], 1e-6)
            r_th = (s * 5.0) ** 2
            k_th = p_th / (p_th + r_th)
            corr_th = k_th * dtheta
            if abs(corr_th) > self.max_rot_correction:
                corr_th = 0.0

            self.pose    = init_pose.copy()
            self.pose[0] += corr_xy[0]
            self.pose[1] += corr_xy[1]
            self.pose[2]  = self._wrap(self.pose[2] + corr_th)

            pts_map = self._rigid2d(pts_robot, *self.pose)
        else:
            self.pose = init_pose.copy()
            self.ekf_cov_2d = self.odom_cov[:2, :2].copy()

        dist_from_kf  = np.linalg.norm(self.pose[:2] - self.ref_pose[:2])
        angle_from_kf = abs(self._wrap(self.pose[2] - self.ref_pose[2]))
        if dist_from_kf >= self.kf_dist or angle_from_kf >= self.kf_angle:
            self.ref_scan   = pts_map
            self.ref_pose   = self.pose.copy()
            self.ref_cov    = self.odom_cov.copy()
            self.ekf_cov_2d = np.eye(2) * 1e-4  # position confirmed — reset to small

        self.scan_count += 1
        if (not self.loop_closed
                and self.scan_count >= self.min_scans_loop
                and self.scan_count % self.loop_check_period == 0):
            self._check_loop_closure(pts_robot)

        self._publish(msg.header.stamp)

    def _check_loop_closure(self, pts_robot: np.ndarray):
        if self.initial_scan is None:
            return
        pts_map = self._rigid2d(pts_robot, *self.pose)
        result  = self._icp(pts_map, self.initial_scan)
        if result is None:
            return

        dx, dy, dtheta, score = result
        self.get_logger().info(
            f'Loop check — score: {score:.3f} m | Δx={dx:.3f} Δy={dy:.3f} Δθ={math.degrees(dtheta):.1f}°')

        if score > self.loop_score_thresh:
            return

        self.get_logger().info(
            f'LOOP CLOSURE — drift corrected: Δx={dx:.3f} Δy={dy:.3f} Δθ={math.degrees(dtheta):.1f}°')
        self.pose[0] += dx
        self.pose[1] += dy
        self.pose[2]  = self._wrap(self.pose[2] + dtheta)
        self.scan_count  = 0
        self.loop_closed = False

    @staticmethod
    def _wrap(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    # ── Publishing ────────────────────────────────────────────────────

    def _publish(self, stamp):
        x, y, th = self.pose
        qz = math.sin(th / 2.0)
        qw = math.cos(th / 2.0)

        ps = PoseStamped()
        ps.header.stamp       = stamp
        ps.header.frame_id    = self.map_frame
        ps.pose.position.x    = x
        ps.pose.position.y    = y
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        self.pose_pub.publish(ps)

        od = Odometry()
        od.header         = ps.header
        od.child_frame_id = self.base_frame
        od.pose.pose      = ps.pose
        # EKF-corrected covariance: shrinks after a good ICP match, grows while drifting
        cov = [0.0] * 36
        cov[0]  = self.ekf_cov_2d[0, 0]   # x-x
        cov[1]  = self.ekf_cov_2d[0, 1]   # x-y
        cov[6]  = self.ekf_cov_2d[1, 0]   # y-x
        cov[7]  = self.ekf_cov_2d[1, 1]   # y-y
        cov[35] = self.odom_cov[2, 2]      # θ-θ from odom (not corrected by ICP)
        od.pose.covariance = cov
        self.odom_pub.publish(od)

        tf = TransformStamped()
        tf.header              = ps.header
        tf.child_frame_id      = self.base_frame + '_icp'
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.rotation.z    = qz
        tf.transform.rotation.w    = qw
        self.tf_br.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = ICPNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
