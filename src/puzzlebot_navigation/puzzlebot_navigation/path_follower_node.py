#!/usr/bin/env python3
# /global_path + /mcl_pose → path_follower → /cmd_vel_reference
# Estados: IDLE → ALIGN (giro en lugar) → DRIVE (Pure Pursuit) → DONE

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped
from std_msgs.msg import Empty

from .utils import quat_to_yaw, wrap_angle


_RELIABLE = QoSProfile(
    depth=5,
    durability=DurabilityPolicy.VOLATILE,
    reliability=ReliabilityPolicy.RELIABLE,
)

# Estados de la máquina de control
_IDLE  = 'IDLE'
_ALIGN = 'ALIGN'
_DRIVE = 'DRIVE'
_DONE  = 'DONE'


class PathFollowerNode(Node):

    def __init__(self):
        super().__init__('path_follower_node')

        self.declare_parameter('lookahead_distance',  0.40)  # [m]
        self.declare_parameter('linear_speed',        0.18)  # [m/s]
        self.declare_parameter('max_angular_speed',   1.20)  # [rad/s]
        self.declare_parameter('goal_tolerance',      0.15)  # [m]
        self.declare_parameter('align_threshold_deg', 35.0)  # entra a ALIGN si error > este
        self.declare_parameter('drive_threshold_deg',  8.0)  # sale de ALIGN si error < este
        self.declare_parameter('speed_curve_gain',     0.50) # reducción de v en curvas
        self.declare_parameter('control_rate',        20.0)  # [Hz]
        self.declare_parameter('stuck_time',           2.5)  # s sin avanzar → re-plan
        self.declare_parameter('stuck_dist',           0.05) # m mínimo para "avanzó"
        self.declare_parameter('max_path_deviation',   0.80) # m → reset seg_idx

        self._L          = self.get_parameter('lookahead_distance').value
        self._v_max      = self.get_parameter('linear_speed').value
        self._w_max      = self.get_parameter('max_angular_speed').value
        self._g_tol      = self.get_parameter('goal_tolerance').value
        self._align_th   = math.radians(self.get_parameter('align_threshold_deg').value)
        self._drive_th   = math.radians(self.get_parameter('drive_threshold_deg').value)
        self._k_curv        = self.get_parameter('speed_curve_gain').value
        ctrl_rate           = self.get_parameter('control_rate').value
        self._stuck_time    = self.get_parameter('stuck_time').value
        self._stuck_dist    = self.get_parameter('stuck_dist').value
        self._max_deviation = self.get_parameter('max_path_deviation').value

        self._wp: list[tuple[float, float]] = []
        self._seg_idx   = 0
        self._state     = _IDLE
        self._rx = self._ry = self._rth = 0.0
        self._pose_ready = False
        self._current_goal: tuple[float, float] | None = None

        self._last_prog_x    = 0.0
        self._last_prog_y    = 0.0
        self._last_prog_time = None   # rclpy.time.Time

        self.create_subscription(Path, '/global_path', self._path_cb, _RELIABLE)
        self.create_subscription(
            PoseWithCovarianceStamped, '/mcl_pose', self._pose_cb, _RELIABLE)
        self.create_subscription(Empty, '/cancel_navigation', self._cancel_cb, 10)
        self.declare_parameter('output_topic', '/cmd_vel_reference')
        out_topic = self.get_parameter('output_topic').value
        self._pub        = self.create_publisher(Twist, out_topic, _RELIABLE)
        self._goal_pub   = self.create_publisher(PoseStamped, '/goal_pose', _RELIABLE)
        self._reached_pub = self.create_publisher(Empty, '/waypoint_reached', _RELIABLE)
        self.get_logger().info(f'output_topic={out_topic}')
        self.create_timer(1.0 / ctrl_rate, self._loop)

        self.get_logger().info(
            f'path_follower listo | L={self._L}m v={self._v_max}m/s '
            f'align>{self.get_parameter("align_threshold_deg").value:.0f}° '
            f'drive<{self.get_parameter("drive_threshold_deg").value:.0f}°')

    def _cancel_cb(self, _msg):
        if self._state not in (_IDLE, _DONE):
            self.get_logger().warn('Navegación cancelada (re-localización activa)')
        self._state        = _IDLE
        self._wp           = []
        self._current_goal = None
        self._stop()

    def _path_cb(self, msg: Path):
        if not msg.poses:
            return
        self._wp = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self._seg_idx = 0
        self._state   = _ALIGN
        gx, gy = self._wp[-1]
        self._current_goal = (gx, gy)
        # Reiniciar detección de atasco
        self._last_prog_x    = self._rx
        self._last_prog_y    = self._ry
        self._last_prog_time = self.get_clock().now()
        self.get_logger().info(
            f'Path recibido: {len(self._wp)} wps | goal=({gx:.2f},{gy:.2f}) | ALIGN')

    def _pose_cb(self, msg: PoseWithCovarianceStamped):
        self._rx  = msg.pose.pose.position.x
        self._ry  = msg.pose.pose.position.y
        self._rth = quat_to_yaw(msg.pose.pose.orientation)
        if not self._pose_ready:
            self._pose_ready = True

    def _loop(self):
        if self._state in (_IDLE, _DONE) or not self._pose_ready:
            self._stop()
            return

        self._check_stuck()

        gx, gy = self._wp[-1]
        dist_goal = math.hypot(gx - self._rx, gy - self._ry)
        if dist_goal < self._g_tol:
            self._state = _DONE
            self._stop()
            self._reached_pub.publish(Empty())
            self.get_logger().info(f'Goal alcanzado | error={dist_goal:.3f} m')
            return

        lx, ly = self._lookahead()

        desired = math.atan2(ly - self._ry, lx - self._rx)
        err     = wrap_angle(desired - self._rth)

        if self._state == _ALIGN:
            self._do_align(err)
        elif self._state == _DRIVE:
            self._do_drive(lx, ly, err, dist_goal)

    def _do_align(self, err: float):
        """
        Rota en el lugar hacia el lookahead.
        Proporcional al error con saturación.
        Sale a DRIVE cuando error < drive_threshold.
        """
        if abs(err) < self._drive_th:
            self._state = _DRIVE
            self.get_logger().info(
                f'Alineado → DRIVE | err={math.degrees(err):.1f}°')
            return

        w = math.copysign(
            min(self._w_max, max(0.3, abs(err)) * 0.8),
            err)
        t = Twist()
        t.angular.z = float(w)
        self._pub.publish(t)

    def _do_drive(self, lx: float, ly: float, err: float, dist_goal: float):
        """
        Pure Pursuit.
        Si el error de rumbo supera align_threshold → vuelve a ALIGN.
        Velocidad adaptativa: se reduce cuanto más cerrada es la curva.
        """
        if abs(err) > self._align_th:
            self._state = _ALIGN
            self.get_logger().info(
                f'Curva cerrada → ALIGN | err={math.degrees(err):.1f}°')
            return

        dx = lx - self._rx
        dy = ly - self._ry
        c, s   = math.cos(self._rth), math.sin(self._rth)
        lx_r   =  dx * c + dy * s    # adelante
        ly_r   = -dx * s + dy * c    # lateral

        L = math.hypot(lx_r, ly_r)
        if L < 1e-4:
            self._stop()
            return

        # Curvatura Pure Pursuit: κ = 2·y_local / L²
        kappa = 2.0 * ly_r / (L * L)
        omega = max(-self._w_max, min(self._w_max, self._v_max * kappa))

        # Velocidad adaptativa: v = v_max / (1 + k·|κ|·L)
        # Ejemplo: kappa=5 rad/m, L=0.4m, k=0.5 → v = v_max/(1+1) = v_max/2
        v = self._v_max / (1.0 + self._k_curv * abs(kappa) * self._L)
        v = max(v, self._v_max * 0.25)  # mínimo 25% de velocidad

        # Reducir al acercarse al goal
        v *= min(1.0, max(0.4, dist_goal / (self._L * 2.0)))

        t = Twist()
        t.linear.x  = float(v)
        t.angular.z = float(omega)
        self._pub.publish(t)

    def _lookahead(self) -> tuple[float, float]:
        """
        Recorre los segmentos del path a partir del segmento activo,
        avanza L metros desde la proyección del robot en el path y
        devuelve el punto resultante.

        Esto hace que el robot siga la LÍNEA del path y no salte de
        waypoint en waypoint.
        """
        wp = self._wp
        n  = len(wp)

        # Si el robot se desvió mucho del path (DWA lo empujó fuera),
        # buscar el segmento más cercano en TODO el path y resetear seg_idx.
        # Evita que el lookahead apunte "hacia atrás" o a un segmento lejano.
        if n > 1 and self._seg_idx < n - 1:
            dist_to_current = math.hypot(
                wp[self._seg_idx][0] - self._rx,
                wp[self._seg_idx][1] - self._ry)
            if dist_to_current > self._max_deviation:
                best_global = min(
                    range(n),
                    key=lambda i: math.hypot(wp[i][0]-self._rx, wp[i][1]-self._ry))
                self._seg_idx = max(0, best_global - 1)
                self.get_logger().info(
                    f'Desviación {dist_to_current:.2f}m → reset seg={self._seg_idx}')

        # Avanzar seg_idx: encontrar el segmento más cercano al robot
        best_d = float('inf')
        for i in range(self._seg_idx, min(n - 1, self._seg_idx + 5)):
            ax, ay = wp[i]
            bx, by = wp[i + 1] if i + 1 < n else wp[i]
            # Proyección del robot en el segmento
            seg_x, seg_y = bx - ax, by - ay
            seg_l2 = seg_x**2 + seg_y**2
            if seg_l2 < 1e-9:
                continue
            t = max(0.0, min(1.0,
                ((self._rx - ax) * seg_x + (self._ry - ay) * seg_y) / seg_l2))
            cx = ax + t * seg_x
            cy = ay + t * seg_y
            d  = math.hypot(cx - self._rx, cy - self._ry)
            if d < best_d:
                best_d = d
                self._seg_idx = i

        # Caminar L metros hacia adelante desde la proyección del robot
        remaining = self._L
        for i in range(self._seg_idx, n - 1):
            ax, ay = wp[i]
            bx, by = wp[i + 1]

            # En el primer segmento, empezar desde la proyección del robot
            if i == self._seg_idx:
                seg_x, seg_y = bx - ax, by - ay
                seg_l2 = seg_x**2 + seg_y**2
                if seg_l2 > 1e-9:
                    t = max(0.0, min(1.0,
                        ((self._rx - ax) * seg_x + (self._ry - ay) * seg_y) / seg_l2))
                    ax = ax + t * seg_x
                    ay = ay + t * seg_y

            seg_len = math.hypot(bx - ax, by - ay)
            if seg_len < 1e-9:
                continue

            if remaining <= seg_len:
                t = remaining / seg_len
                return ax + t * (bx - ax), ay + t * (by - ay)
            remaining -= seg_len

        return wp[-1]

    def _check_stuck(self):
        """
        Detecta si el robot lleva más de stuck_time segundos sin avanzar
        stuck_dist metros. Si es así, re-publica el mismo goal al RRT
        para que replantee la ruta.
        """
        if self._state in (_IDLE, _DONE) or self._last_prog_time is None:
            return

        dist_moved = math.hypot(
            self._rx - self._last_prog_x,
            self._ry - self._last_prog_y)

        if dist_moved >= self._stuck_dist:
            self._last_prog_x    = self._rx
            self._last_prog_y    = self._ry
            self._last_prog_time = self.get_clock().now()
            return

        elapsed = (self.get_clock().now() - self._last_prog_time).nanoseconds / 1e9
        if elapsed >= self._stuck_time:
            self.get_logger().warn(
                f'Robot atascado {elapsed:.1f}s sin avanzar → re-planificando')
            self._trigger_replan()
            self._last_prog_time = self.get_clock().now()

    def _trigger_replan(self):
        """Re-publica el goal actual para que el RRT genere un nuevo path."""
        if self._current_goal is None:
            return
        gx, gy = self._current_goal
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = gx
        msg.pose.position.y = gy
        msg.pose.orientation.w = 1.0
        self._goal_pub.publish(msg)
        self.get_logger().info(f'Re-plan → goal=({gx:.2f},{gy:.2f})')

    def _stop(self):
        self._pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = PathFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
