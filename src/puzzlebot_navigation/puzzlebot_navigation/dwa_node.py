#!/usr/bin/env python3
# /cmd_vel_reference + /scan → dwa_node → /cmd_vel
# Samplea (v,ω) en la ventana dinámica, simula trayectorias y publica la más segura.

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from rcl_interfaces.msg import SetParametersResult

from .utils import quat_to_yaw, wrap_angle


_RELIABLE = QoSProfile(
    depth=5,
    durability=DurabilityPolicy.VOLATILE,
    reliability=ReliabilityPolicy.RELIABLE,
)


class DWANode(Node):

    def __init__(self):
        super().__init__('dwa_node')

        self.declare_parameter('v_max',          0.20)   # velocidad lineal máxima [m/s]
        self.declare_parameter('omega_max',       1.20)  # velocidad angular máxima [rad/s]
        self.declare_parameter('accel_v',         0.80)  # aceleración lineal [m/s²]
        self.declare_parameter('accel_omega',     1.60)  # aceleración angular [rad/s²]
        # robot_radius DEBE ser < inflation_radius del costmap.
        # El path ya tiene ese margen; aquí solo protegemos obstáculos dinámicos.
        self.declare_parameter('robot_radius',    0.16)  # radio del robot + margen [m]
        self.declare_parameter('sim_time',        1.0)   # tiempo de simulación [s]
        self.declare_parameter('sim_dt',          0.10)  # paso de simulación [s]
        self.declare_parameter('v_samples',       8)     # muestras lineales
        self.declare_parameter('omega_samples',   16)    # muestras angulares
        self.declare_parameter('w_heading',       0.50)  # peso: apuntar al goal
        self.declare_parameter('w_clearance',     0.30)  # peso: distancia a obstáculos
        self.declare_parameter('w_velocity',      0.20)  # peso: seguir velocidad ref.
        self.declare_parameter('lookahead',       0.50)  # distancia lookahead [m]
        self.declare_parameter('goal_tolerance',  0.15)  # radio de llegada [m]
        self.declare_parameter('scan_topic',      '/scan')
        self.declare_parameter('control_rate',    20.0)  # Hz

        self._v_max       = self.get_parameter('v_max').value
        self._w_max       = self.get_parameter('omega_max').value
        self._a_v         = self.get_parameter('accel_v').value
        self._a_w         = self.get_parameter('accel_omega').value
        self._r_robot     = self.get_parameter('robot_radius').value
        self._sim_t       = self.get_parameter('sim_time').value
        self._sim_dt      = self.get_parameter('sim_dt').value
        self._n_v         = self.get_parameter('v_samples').value
        self._n_w         = self.get_parameter('omega_samples').value
        self._wh          = self.get_parameter('w_heading').value
        self._wd          = self.get_parameter('w_clearance').value
        self._wv          = self.get_parameter('w_velocity').value
        self._lookahead   = self.get_parameter('lookahead').value
        self._g_tol       = self.get_parameter('goal_tolerance').value
        scan_topic        = self.get_parameter('scan_topic').value
        ctrl_rate         = self.get_parameter('control_rate').value

        self._n_steps = max(2, int(self._sim_t / self._sim_dt))
        self._t_arr   = np.linspace(self._sim_dt, self._sim_t, self._n_steps)

        # Puntos del scan en frame robot: array (N, 2) con (x, y) en metros
        self._scan_xy: np.ndarray | None = None

        self._curr_v = 0.0
        self._curr_w = 0.0

        self._ref_v = 0.0
        self._ref_w = 0.0
        self._ref_updated = False   # True cuando llega /cmd_vel_reference

        self._wp: list[tuple[float, float]] = []
        self._pose_ready = False
        self._rx = self._ry = self._rth = 0.0
        self._wandering = False     # True cuando active_localization controla cmd_vel
        self._external_active = False  # True cuando otro nodo (p.ej. mission_manager
                                        # durante el barrido) controla /cmd_vel directo

        self.create_subscription(
            LaserScan, scan_topic, self._scan_cb, _RELIABLE)
        self.create_subscription(
            Twist, '/cmd_vel_reference', self._ref_cb, _RELIABLE)
        self.create_subscription(
            Path, '/global_path', self._path_cb, _RELIABLE)
        self.create_subscription(
            PoseWithCovarianceStamped, '/mcl_pose', self._pose_cb, _RELIABLE)
        self.create_subscription(
            Bool, '/mcl_wandering', self._wandering_cb, 10)
        self.create_subscription(
            Bool, '/external_cmd_vel_active', self._external_cmd_cb, 10)

        self._pub = self.create_publisher(Twist, '/cmd_vel', _RELIABLE)
        self.create_timer(1.0 / ctrl_rate, self._loop)
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.get_logger().info(
            f'dwa_node listo | r_robot={self._r_robot} m | '
            f'sim={self._sim_t}s/{self._sim_dt}s | '
            f'samples={self._n_v}×{self._n_w}={self._n_v*self._n_w} | '
            f'scan={scan_topic}')

    def _wandering_cb(self, msg: Bool):
        self._wandering = msg.data

    def _external_cmd_cb(self, msg: Bool):
        self._external_active = msg.data

    def _on_set_parameters(self, params):
        """Permite cambiar 'robot_radius' en caliente (mission_manager lo
        reduce al entrar a la zona de entrega y lo restaura al salir)."""
        for p in params:
            if p.name == 'robot_radius':
                self._r_robot = float(p.value)
                self.get_logger().info(f'robot_radius actualizado a {self._r_robot} m')
        return SetParametersResult(successful=True)

    def _scan_cb(self, msg: LaserScan):
        """Convierte el LaserScan a puntos (x,y) en el frame del robot."""
        n      = len(msg.ranges)
        angles = msg.angle_min + np.arange(n) * msg.angle_increment
        ranges = np.array(msg.ranges, dtype=np.float32)

        valid  = (np.isfinite(ranges) &
                  (ranges >= msg.range_min) &
                  (ranges <= msg.range_max))
        r = ranges[valid].astype(np.float64)
        a = angles[valid]

        self._scan_xy = np.column_stack([r * np.cos(a), r * np.sin(a)])

    def _ref_cb(self, msg: Twist):
        """Velocidad de referencia publicada por path_follower."""
        self._ref_v       = msg.linear.x
        self._ref_w       = msg.angular.z
        self._ref_updated = True

    def _path_cb(self, msg: Path):
        self._wp = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def _pose_cb(self, msg: PoseWithCovarianceStamped):
        self._rx  = msg.pose.pose.position.x
        self._ry  = msg.pose.pose.position.y
        self._rth = quat_to_yaw(msg.pose.pose.orientation)
        if not self._pose_ready:
            self._pose_ready = True

    def _loop(self):
        # Ceder control a active_localization_node durante re-localización activa
        if self._wandering:
            return

        # Ceder control a otro nodo que está manejando /cmd_vel directamente
        # (p.ej. mission_manager_node durante el barrido rotacional de pallets)
        if self._external_active:
            return

        # Sin datos suficientes → parar (seguro por defecto)
        if self._scan_xy is None or not self._ref_updated or not self._pose_ready:
            self._pub_stop()
            return

        # Si la referencia es cero (path_follower paró) → parar
        if abs(self._ref_v) < 1e-4 and abs(self._ref_w) < 1e-4:
            self._curr_v = 0.0
            self._curr_w = 0.0
            self._pub_stop()
            return

        # Freno de emergencia: si el freno de proximidad ya redujo la
        # velocidad a 0 pero path_follower pide avanzar → recuperación directa
        if self._ref_v > 0.02:
            braked_check = self._proximity_brake(self._ref_v, self._scan_xy)
            if braked_check < 0.01:
                best_v, best_w = self._recovery(self._scan_xy)
                self._curr_v = best_v
                self._curr_w = best_w
                t = Twist()
                t.linear.x  = float(best_v)
                t.angular.z = float(best_w)
                self._pub.publish(t)
                return

        # Bypass: rotación en lugar (path_follower en ALIGN, v≈0) -- la
        # trayectoria no avanza, el DWA no debe interferir.
        if abs(self._ref_v) < 0.02 and abs(self._ref_w) > 0.05:
            t = Twist()
            t.angular.z = float(self._ref_w)
            self._pub.publish(t)
            return

        # Centrar muestras angulares alrededor de la referencia para
        # que el DWA prefiera naturalmente la trayectoria del path_follower.
        w_center = float(np.clip(self._ref_w, -self._w_max, self._w_max))
        w_half   = self._w_max
        v_arr = np.linspace(0.0,            self._v_max,          self._n_v)
        w_arr = np.linspace(w_center - w_half, w_center + w_half, self._n_w)
        w_arr = np.clip(w_arr, -self._w_max, self._w_max)

        ref_v_braked = self._proximity_brake(self._ref_v, self._scan_xy)

        ref_traj, _ = self._simulate(ref_v_braked, self._ref_w)
        if self._clearance(ref_traj, self._scan_xy) >= self._r_robot:
            self._curr_v = ref_v_braked
            self._curr_w = self._ref_w
            t = Twist()
            t.linear.x  = float(ref_v_braked)
            t.angular.z = float(self._ref_w)
            self._pub.publish(t)
            return

        goal_heading_local = self._goal_heading_local()

        best_score = -np.inf
        best_v, best_w = 0.0, 0.0
        found_safe   = False

        scan = self._scan_xy   # (N, 2) fijo durante todo el loop

        for v in v_arr:
            for w in w_arr:
                traj_xy, final_th = self._simulate(v, w)
                min_dist = self._clearance(traj_xy, scan)

                if min_dist < self._r_robot:
                    continue

                h_score = 1.0 - abs(wrap_angle(goal_heading_local - final_th)) / math.pi
                d_score = min(1.0, min_dist / (self._v_max * self._sim_t))
                v_score = 1.0 - abs(v - self._ref_v) / max(self._v_max, 1e-3)

                score = self._wh * h_score + self._wd * d_score + self._wv * v_score

                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w
                    found_safe = True

        if not found_safe:
            best_v, best_w = self._recovery(scan)

        self._curr_v = best_v
        self._curr_w = best_w

        t = Twist()
        t.linear.x  = float(best_v)
        t.angular.z = float(best_w)
        self._pub.publish(t)

    def _simulate(self, v: float, w: float):
        """Simula una trayectoria circular en frame LOCAL del robot.
        Retorna (traj_xy (n_steps,2), final_th)."""
        t = self._t_arr  # (n_steps,)
        if abs(w) > 1e-4:
            x = (v / w) * np.sin(w * t)
            y = (v / w) * (1.0 - np.cos(w * t))
        else:
            x = v * t
            y = np.zeros_like(t)
        final_th = w * self._sim_t
        return np.column_stack([x, y]), final_th

    def _clearance(self, traj_xy: np.ndarray,
                   scan_xy: np.ndarray) -> float:
        """Distancia mínima entre la trayectoria y los puntos del scan."""
        if len(scan_xy) == 0:
            return float('inf')

        # Broadcasting: (n_steps, 1, 2) - (1, N_scan, 2) → (n_steps, N_scan)
        diff  = traj_xy[:, np.newaxis, :] - scan_xy[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        return float(dists.min())

    def _goal_heading_local(self) -> float:
        """Ángulo hacia el lookahead del path global en frame LOCAL (0 si no hay path)."""
        if not self._wp or not self._pose_ready:
            return 0.0

        # Encontrar lookahead a self._lookahead metros
        for wx, wy in self._wp:
            d = math.hypot(wx - self._rx, wy - self._ry)
            if d >= self._lookahead:
                # Transformar al frame del robot
                dx = wx - self._rx
                dy = wy - self._ry
                # Ángulo en frame robot
                heading_map  = math.atan2(dy, dx)
                heading_local = wrap_angle(heading_map - self._rth)
                return heading_local

        # Si todos los waypoints están más cerca que lookahead → apuntar al último
        wx, wy = self._wp[-1]
        heading_map   = math.atan2(wy - self._ry, wx - self._rx)
        return wrap_angle(heading_map - self._rth)

    def _proximity_brake(self, ref_v: float, scan_xy: np.ndarray) -> float:
        """Reduce la velocidad lineal según cercanía del obstáculo más cercano
        en el cono frontal: full speed >= brake_dist, 0 si <= robot_radius,
        interpolación lineal entre ambos."""
        if len(scan_xy) == 0 or ref_v <= 0:
            return ref_v

        x_pts = scan_xy[:, 0]
        y_pts = scan_xy[:, 1]

        # ±60° en lugar de ±30° — los cilindros a 40-50° no se detectaban con el cono estrecho
        angles    = np.arctan2(y_pts, x_pts)
        en_frente = (x_pts > 0) & (np.abs(angles) < math.radians(60))

        if not np.any(en_frente):
            return ref_v

        min_fwd = float(np.min(np.linalg.norm(scan_xy[en_frente], axis=1)))

        brake_dist = 0.60   # empieza a frenar a 60cm del obstáculo

        if min_fwd >= brake_dist:
            return ref_v
        if min_fwd <= self._r_robot:
            return 0.0

        factor = (min_fwd - self._r_robot) / (brake_dist - self._r_robot)
        braked = ref_v * max(0.0, min(1.0, factor))
        return braked

    def _recovery(self, scan_xy: np.ndarray) -> tuple[float, float]:
        """Sin trayectoria segura: gira hacia el lado con más espacio (frame
        local x=adelante, y=izquierda); si ambos bloqueados, reversa si hay
        espacio atrás; si no, stop y esperar re-plan."""
        if len(scan_xy) == 0:
            # Sin datos de scan: girar a la izquierda por defecto
            return 0.0, self._w_max * 0.5

        x_pts = scan_xy[:, 0]
        y_pts = scan_xy[:, 1]

        def sector_clearance(mask):
            """Distancia mínima a los puntos del sector indicado."""
            pts = scan_xy[mask]
            if len(pts) == 0:
                return float('inf')
            return float(np.min(np.linalg.norm(pts, axis=1)))

        # Solo puntos del FRENTE para decidir dirección de giro
        frente = x_pts > 0.0

        c_izq   = sector_clearance(frente & (y_pts >  0.05))   # frente-izquierda
        c_der   = sector_clearance(frente & (y_pts < -0.05))   # frente-derecha
        c_atras = sector_clearance(x_pts < -0.05)               # detrás (completo)

        umbral_bloqueado = self._r_robot * 1.5   # considera "bloqueado" si < 1.5×r
        umbral_reversa   = self._r_robot * 3.0   # reversa solo si atrás tiene margen

        izq_libre   = c_izq   > umbral_bloqueado
        der_libre   = c_der   > umbral_bloqueado
        atras_libre = c_atras > umbral_reversa

        w_rec = self._w_max * 0.5   # velocidad de giro de recuperación

        if izq_libre and c_izq >= c_der:
            self.get_logger().info(
                f'Recuperación: giro izq | c_izq={c_izq:.2f} c_der={c_der:.2f}',
                throttle_duration_sec=1.0)
            return 0.0, w_rec

        if der_libre and c_der > c_izq:
            self.get_logger().info(
                f'Recuperación: giro der | c_izq={c_izq:.2f} c_der={c_der:.2f}',
                throttle_duration_sec=1.0)
            return 0.0, -w_rec

        if atras_libre:
            self.get_logger().info(
                f'Recuperación: reversa | c_atras={c_atras:.2f}',
                throttle_duration_sec=1.0)
            return -self._v_max * 0.35, 0.0

        # Todo bloqueado → stop y esperar que path_follower reciba un nuevo path
        self.get_logger().warn('Recuperación: todo bloqueado → stop',
                               throttle_duration_sec=2.0)
        return 0.0, 0.0

    def _pub_stop(self):
        self._pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = DWANode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
