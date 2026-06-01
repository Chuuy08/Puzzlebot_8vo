#!/usr/bin/env python3
import math
import random
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from std_msgs.msg import Bool, Empty
from .utils import quat_to_yaw, wrap_angle


_WANDERING = 'WANDERING'
_LOCALIZED = 'LOCALIZED'


class ActiveLocalizationNode(Node):
    """
    Nodo de localización activa para simulación.

    WANDERING: mueve el robot con evasión reactiva de obstáculos e inyección
               de giros aleatorios para que MCL acumule odometría informativa
               y converja más rápido.
    LOCALIZED: para publicar cmd_vel y cede el control al stack de navegación
               (DWA + path_follower).  El robot espera un 2D Goal Pose en RViz.

    Transitions:
      WANDERING → LOCALIZED  cuando /mcl_converged=True durante >= convergence_hold_time
      LOCALIZED → WANDERING  cuando /mcl_converged=False (MCL perdió localización)
                             → publica /cancel_navigation para resetear path_follower

    El topic /mcl_wandering (Bool, latch) indica a dwa_node que ceda /cmd_vel.
    Un heartbeat a 2 Hz garantiza que nuevos subscribers reciban el estado actual.
    Un timeout configurable emite warnings si MCL no converge a tiempo, pero
    el robot sigue wandering — no se detiene solo por timeout.
    """

    def __init__(self):
        super().__init__('active_localization_node')

        self.declare_parameter('scan_topic',           '/scan')
        self.declare_parameter('wander_timeout',        120.0)   # [s] límite de seguridad
        self.declare_parameter('wander_linear_speed',     0.10)  # [m/s]
        self.declare_parameter('wander_angular_speed',    0.50)  # [rad/s]
        self.declare_parameter('obstacle_distance',       0.40)  # [m] umbral frente
        self.declare_parameter('random_turn_interval',    5.0)   # [s] entre giros aleatorios
        self.declare_parameter('convergence_hold_time',   5.0)   # [s] MCL estable mínimo
        self.declare_parameter('min_conv_travel_m',       0.20)  # [m] recorrido mínimo durante convergencia
        self.declare_parameter('min_conv_rotation_deg',  90.0)   # [°] rotación mínima acumulada durante convergencia
        self.declare_parameter('front_half_angle_deg',   60.0)   # [°] semi-ángulo sector frontal

        scan_topic        = self.get_parameter('scan_topic').value
        self._timeout     = self.get_parameter('wander_timeout').value
        self._lin_spd     = self.get_parameter('wander_linear_speed').value
        self._ang_spd     = self.get_parameter('wander_angular_speed').value
        self._obs_dist    = self.get_parameter('obstacle_distance').value
        self._rand_ivl    = self.get_parameter('random_turn_interval').value
        self._conv_hold   = self.get_parameter('convergence_hold_time').value
        self._min_travel  = self.get_parameter('min_conv_travel_m').value
        self._min_rot_rad = math.radians(self.get_parameter('min_conv_rotation_deg').value)
        self._front_ang   = math.radians(self.get_parameter('front_half_angle_deg').value)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self._scan: LaserScan | None = None
        self._state        = _WANDERING
        self._mcl_conv     = False
        self._conv_since   = None     # tiempo en que MCL reportó convergencia por primera vez
        self._wander_start = self._now()

        # Evidencia de movimiento acumulada mientras MCL reporta convergencia.
        # Exigir traslación >= min_conv_travel_m Y rotación >= min_conv_rotation_deg
        # antes de declarar LOCALIZED.
        #
        # La rotación es el discriminador clave contra ambigüedad de 180°:
        # si el robot está rotado 180° respecto a la posición real, al girar ~90°
        # los obstáculos internos del mapa aparecerán en el lado equivocado del scan
        # → MCL pierde convergencia → ambos contadores se reinician.
        self._travel_while_conv  = 0.0
        self._rotation_while_conv = 0.0
        self._pose_x: float | None = None
        self._pose_y: float | None = None
        self._prev_yaw: float | None = None

        # Estado para giros aleatorios
        self._turn_dir    = 1.0
        self._turn_until  = 0.0
        self._next_turn   = self._now() + random.uniform(2.0, self._rand_ivl)

        self.create_subscription(LaserScan, scan_topic, self._scan_cb, sensor_qos)
        self.create_subscription(Bool, '/mcl_converged', self._conv_cb, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, '/mcl_pose', self._pose_cb, 10)

        self._cmd_pub    = self.create_publisher(Twist, '/cmd_vel',           10)
        self._wand_pub   = self.create_publisher(Bool,  '/mcl_wandering',     10)
        self._cancel_pub = self.create_publisher(Empty, '/cancel_navigation', 10)

        self.create_timer(0.10, self._control_loop)   # 10 Hz — control de wandering
        self.create_timer(0.50, self._heartbeat)      # 2 Hz — mantener estado en /mcl_wandering

        self._set_wandering(True)
        self.get_logger().info(
            f'ActiveLocalization | WANDERING | timeout={self._timeout:.0f}s | '
            f'v={self._lin_spd} m/s  ω={self._ang_spd} rad/s | '
            f'criterio: {self._conv_hold}s + {self._min_travel}m + '
            f'{math.degrees(self._min_rot_rad):.0f}° rotación')

    # ── helpers ───────────────────────────────────────────────────────────────

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _set_wandering(self, value: bool):
        self._wand_pub.publish(Bool(data=value))

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _scan_cb(self, msg: LaserScan):
        self._scan = msg

    def _pose_cb(self, msg: PoseWithCovarianceStamped):
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        yaw = quat_to_yaw(msg.pose.pose.orientation)

        if self._mcl_conv:
            if self._pose_x is not None:
                self._travel_while_conv += math.hypot(x - self._pose_x, y - self._pose_y)
            if self._prev_yaw is not None:
                self._rotation_while_conv += abs(wrap_angle(yaw - self._prev_yaw))

        self._pose_x   = x
        self._pose_y   = y
        self._prev_yaw = yaw

    def _conv_cb(self, msg: Bool):
        now = self._now()
        if msg.data and not self._mcl_conv:
            self._mcl_conv            = True
            self._conv_since          = now
            self._travel_while_conv   = 0.0
            self._rotation_while_conv = 0.0
            self.get_logger().info(
                f'MCL convergido — necesito {self._conv_hold:.1f}s + '
                f'{self._min_travel:.2f}m + {math.degrees(self._min_rot_rad):.0f}° para confirmar...')
        elif not msg.data and self._mcl_conv:
            self._mcl_conv            = False
            self._conv_since          = None
            self._travel_while_conv   = 0.0
            self._rotation_while_conv = 0.0
            if self._state == _LOCALIZED:
                self._to_wandering()

    # ── transiciones de estado ────────────────────────────────────────────────

    def _to_wandering(self):
        self._state               = _WANDERING
        self._wander_start        = self._now()
        self._travel_while_conv   = 0.0
        self._rotation_while_conv = 0.0
        self._set_wandering(True)
        self._cancel_pub.publish(Empty())
        self._cmd_pub.publish(Twist())
        self.get_logger().warn(
            'Localización perdida → WANDERING | navegación cancelada')

    def _to_localized(self):
        self._state = _LOCALIZED
        self._set_wandering(False)
        self._cmd_pub.publish(Twist())
        self.get_logger().info(
            f'MCL confirmado ({self._travel_while_conv:.2f}m | '
            f'{math.degrees(self._rotation_while_conv):.0f}° rotados) → LOCALIZED | '
            f'envía un 2D Goal Pose en RViz para navegar')

    # ── timers ────────────────────────────────────────────────────────────────

    def _heartbeat(self):
        self._set_wandering(self._state == _WANDERING)
        if self._state == _WANDERING:
            elapsed = self._now() - self._wander_start
            if elapsed > self._timeout:
                self.get_logger().warn(
                    f'Wandering timeout ({self._timeout:.0f}s) sin convergencia — '
                    f'verifica el mapa, el LiDAR y los parámetros MCL.',
                    throttle_duration_sec=10.0)

    def _control_loop(self):
        now = self._now()

        # Transición WANDERING → LOCALIZED: los tres criterios deben cumplirse simultáneamente:
        #   1. Tiempo: MCL reporta convergencia >= hold_time segundos sin interrupción
        #   2. Traslación: robot recorrió >= min_travel metros mientras convergía
        #   3. Rotación: robot rotó >= min_rot_rad acumulados mientras convergía
        #
        # La rotación es el discriminador clave para la ambigüedad de 180°:
        # en la orientación incorrecta, rotar ~90° hace que los obstáculos internos
        # aparezcan en el lado equivocado del scan → MCL pierde convergencia → reset.
        if self._state == _WANDERING:
            if (self._mcl_conv
                    and self._conv_since is not None
                    and now - self._conv_since          >= self._conv_hold
                    and self._travel_while_conv         >= self._min_travel
                    and self._rotation_while_conv       >= self._min_rot_rad):
                self._to_localized()
                return

        if self._state == _LOCALIZED:
            return

        if self._scan is None:
            return   # esperar primer scan

        self._cmd_pub.publish(self._wander_cmd(now))

    # ── planificador de wandering ─────────────────────────────────────────────

    def _wander_cmd(self, now: float) -> Twist:
        scan    = self._scan
        ranges  = np.asarray(scan.ranges, dtype=np.float64)
        angles  = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        valid   = (np.isfinite(ranges)
                   & (ranges > scan.range_min)
                   & (ranges < scan.range_max))
        max_r   = scan.range_max if scan.range_max > 0.1 else 10.0
        safe_r  = np.where(valid, ranges, max_r)

        fa = self._front_ang
        front_m = np.abs(angles) < fa
        left_m  = (angles >=  fa) & (angles <  math.pi - 0.05)
        right_m = (angles <= -fa) & (angles > -(math.pi - 0.05))

        front_min = float(np.min(safe_r[front_m])) if np.any(front_m) else max_r
        left_min  = float(np.min(safe_r[left_m]))  if np.any(left_m)  else max_r
        right_min = float(np.min(safe_r[right_m])) if np.any(right_m) else max_r

        cmd = Twist()

        # ── Giro aleatorio activo ─────────────────────────────────────────────
        # Genera odometría rotacional que MCL necesita para discriminar orientación.
        if now < self._turn_until:
            cmd.angular.z = self._turn_dir * self._ang_spd
            return cmd

        if now >= self._next_turn:
            self._turn_dir   = 1.0 if random.random() > 0.5 else -1.0
            dur              = random.uniform(0.6, 1.8)
            self._turn_until = now + dur
            self._next_turn  = now + dur + self._rand_ivl + random.uniform(-1.0, 1.0)
            cmd.angular.z    = self._turn_dir * self._ang_spd
            return cmd

        # ── Evasión reactiva de obstáculos ────────────────────────────────────
        if front_min < self._obs_dist:
            # Girar hacia el lado más despejado
            cmd.angular.z = self._ang_spd if left_min >= right_min else -self._ang_spd
        else:
            cmd.linear.x = self._lin_spd
            # Sesgo suave hacia el espacio más abierto para favorecer exploración
            if left_min > right_min + 0.30:
                cmd.angular.z =  0.15
            elif right_min > left_min + 0.30:
                cmd.angular.z = -0.15

        return cmd


def main(args=None):
    rclpy.init(args=args)
    node = ActiveLocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
