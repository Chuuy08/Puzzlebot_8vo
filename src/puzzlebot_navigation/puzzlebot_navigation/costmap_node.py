#!/usr/bin/env python3
# /map + /scan + /mcl_pose → costmap_node → /costmap
# Costmap: 0=libre, 99=inflado, 100=obstáculo, -1=desconocido

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
from rcl_interfaces.msg import SetParametersResult

from .utils import quat_to_yaw


# QoS latched: el suscriptor recibe el último mensaje aunque arranque tarde.
# Obligatorio para /map y /costmap (se publican una sola vez al inicio).
_LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)

# QoS estándar para topics que llegan continuamente (scan, pose).
_RELIABLE_QOS = QoSProfile(
    depth=5,
    durability=DurabilityPolicy.VOLATILE,
    reliability=ReliabilityPolicy.RELIABLE,
)


class CostmapNode(Node):

    def __init__(self):
        super().__init__('costmap_node')

        self.declare_parameter('inflation_radius',         0.18)   # radio inflado estático [m]
        self.declare_parameter('dynamic_inflation_radius', 0.18)   # radio inflado dinámico [m]
        self.declare_parameter('publish_rate',             5.0)    # Hz
        # scan_topic: '/scan' en simulación, '/scan_fixed' en robot real
        self.declare_parameter('scan_topic',               '/scan')
        # laser_angle_offset: 0.0 sim (Gazebo angle=0 = adelante),
        #                     π   real (cable RPLidar apunta hacia atrás)
        self.declare_parameter('laser_angle_offset',       0.0)
        # Posición del LiDAR respecto a base_footprint (de URDF: lidar_base_pos_x)
        self.declare_parameter('laser_x_offset',           0.0425) # [m]
        self.declare_parameter('laser_y_offset',           0.0)    # [m]
        self.declare_parameter('laser_max_range',          4.5)    # [m]
        self.declare_parameter('laser_min_range',          0.15)   # [m]

        self._r_inf        = self.get_parameter('inflation_radius').value
        self._dyn_r_inf    = self.get_parameter('dynamic_inflation_radius').value
        pub_rate           = self.get_parameter('publish_rate').value
        scan_topic         = self.get_parameter('scan_topic').value
        self._laser_offset = self.get_parameter('laser_angle_offset').value
        self._lx           = self.get_parameter('laser_x_offset').value
        self._ly           = self.get_parameter('laser_y_offset').value
        self._rmax         = self.get_parameter('laser_max_range').value
        self._rmin         = self.get_parameter('laser_min_range').value

        self._static_costmap: np.ndarray | None = None
        self._raw_map: np.ndarray | None = None   # capa de /map sin inflar — se cachea
                                                   # para poder recalcular _static_costmap
                                                   # si 'inflation_radius' cambia en caliente
                                                   # (ver _on_set_parameters, usado por
                                                   # mission_manager para desinflar/inflar
                                                   # alrededor de la zona de entrega)

        self._map_W   = 0
        self._map_H   = 0
        self._map_res = 0.05
        self._map_ox  = 0.0    # origen x del mapa en frame world
        self._map_oy  = 0.0    # origen y del mapa en frame world
        self._map_cos = 1.0    # cos(yaw_mapa)
        self._map_sin = 0.0    # sin(yaw_mapa)
        self._map_info = None  # MapMetaData completo para el mensaje de salida

        self._dynamic_grid: np.ndarray | None = None

        self._robot_x   = 0.0
        self._robot_y   = 0.0
        self._robot_yaw = 0.0
        self._pose_ready = False   # True una vez que llegó la primera pose

        self._costmap_msg: OccupancyGrid | None = None

        self._map_sub = self.create_subscription(
            OccupancyGrid, '/map', self._map_cb, _LATCHED_QOS)

        self._scan_sub = self.create_subscription(
            LaserScan, scan_topic, self._scan_cb, _RELIABLE_QOS)

        self._pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/mcl_pose', self._pose_cb, _RELIABLE_QOS)

        self._pub = self.create_publisher(OccupancyGrid, '/costmap', _LATCHED_QOS)

        self.create_timer(1.0 / pub_rate, self._timer_cb)
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.get_logger().info(
            f'costmap_node listo | inflation={self._r_inf} m '
            f'| scan_topic={scan_topic} '
            f'| laser_offset={math.degrees(self._laser_offset):.1f}° '
            f'| rate={pub_rate} Hz | esperando /map y /mcl_pose ...')

    def _map_cb(self, msg: OccupancyGrid):
        W   = msg.info.width
        H   = msg.info.height
        res = msg.info.resolution

        self._map_W   = W
        self._map_H   = H
        self._map_res = res
        self._map_ox  = msg.info.origin.position.x
        self._map_oy  = msg.info.origin.position.y
        self._map_info = msg.info

        map_yaw       = quat_to_yaw(msg.info.origin.orientation)
        self._map_cos = math.cos(map_yaw)
        self._map_sin = math.sin(map_yaw)

        raw = np.array(msg.data, dtype=np.int8).reshape(H, W)
        self._raw_map = raw
        self._static_costmap = self._inflate(raw, res, self._r_inf)
        self._dynamic_grid = np.zeros((H, W), dtype=np.int8)
        self._publish_combined()

        n_par = int(np.sum(raw                 == 100))
        n_inf = int(np.sum(self._static_costmap == 99))
        n_lib = int(np.sum(self._static_costmap ==  0))
        self.get_logger().info(
            f'Mapa cargado: {W}×{H} @ {res} m/px | '
            f'paredes={n_par} infladas={n_inf} libres={n_lib}')

    def _on_set_parameters(self, params):
        """Permite cambiar 'inflation_radius' en caliente (servicio
        set_parameters) y recalcula _static_costmap al vuelo a partir del
        _raw_map cacheado — por defecto el valor se lee una sola vez en
        __init__ y queda congelado. mission_manager usa esto para "desinflar"
        la zona de entrega (que de otro modo queda dentro del costmap inflado
        y rrt_node nunca encuentra ruta) y volver a inflarla al terminar."""
        for p in params:
            if p.name == 'inflation_radius':
                self._r_inf = float(p.value)
                if self._raw_map is not None:
                    self._static_costmap = self._inflate(self._raw_map, self._map_res, self._r_inf)
                    self._publish_combined()
                    n_inf = int(np.sum(self._static_costmap == 99))
                    n_lib = int(np.sum(self._static_costmap == 0))
                    self.get_logger().info(
                        f'inflation_radius actualizado a {self._r_inf} m | '
                        f'infladas={n_inf} libres={n_lib}')
            elif p.name == 'dynamic_inflation_radius':
                # A diferencia de la estática, la capa dinámica se recalcula
                # de cero en cada /scan (_scan_cb -> _inflate_hits) usando
                # self._dyn_r_inf directo — no hay nada cacheado que recalcular,
                # el cambio queda activo desde el siguiente scan.
                self._dyn_r_inf = float(p.value)
                self.get_logger().info(f'dynamic_inflation_radius actualizado a {self._dyn_r_inf} m')
        return SetParametersResult(successful=True)

    def _pose_cb(self, msg: PoseWithCovarianceStamped):
        self._robot_x   = msg.pose.pose.position.x
        self._robot_y   = msg.pose.pose.position.y
        self._robot_yaw = quat_to_yaw(msg.pose.pose.orientation)

        if not self._pose_ready:
            self._pose_ready = True
            self.get_logger().info(
                f'Primera pose recibida: '
                f'({self._robot_x:.2f}, {self._robot_y:.2f}) '
                f'yaw={math.degrees(self._robot_yaw):.1f}° — capa dinámica activa')

    def _scan_cb(self, msg: LaserScan):
        if self._static_costmap is None:
            return
        if not self._pose_ready:
            self.get_logger().warn(
                'Scan recibido pero sin pose del robot aún. '
                'Haz clic en "2D Pose Estimate" en RViz para activar la capa dinámica.',
                throttle_duration_sec=5.0)
            return

        n      = len(msg.ranges)
        ranges = np.array(msg.ranges, dtype=np.float64)

        angles_local = msg.angle_min + np.arange(n) * msg.angle_increment

        valido = (
            np.isfinite(ranges) &
            (ranges >= self._rmin) &
            (ranges <  self._rmax)
        )
        r = ranges[valido]
        a = angles_local[valido]

        if len(r) == 0:
            return

        # Posición del LiDAR en el frame map
        # (el LiDAR está desplazado lx adelante, ly lateral respecto a base_footprint)
        cos_r = math.cos(self._robot_yaw)
        sin_r = math.sin(self._robot_yaw)
        laser_x = self._robot_x + self._lx * cos_r - self._ly * sin_r
        laser_y = self._robot_y + self._lx * sin_r + self._ly * cos_r

        # Ángulo de cada beam en el frame map
        # laser_angle_offset corrige la orientación física del sensor
        beam_angles = self._robot_yaw + a + self._laser_offset

        hit_x = laser_x + r * np.cos(beam_angles)
        hit_y = laser_y + r * np.sin(beam_angles)

        cols, rows = self._world_to_cell_v(hit_x, hit_y)

        in_bounds = (
            (cols >= 0) & (cols < self._map_W) &
            (rows >= 0) & (rows < self._map_H)
        )
        cols = cols[in_bounds]
        rows = rows[in_bounds]

        # Se limpia en cada scan para eliminar obstáculos que ya no están.
        self._dynamic_grid[:] = 0

        if self._dyn_r_inf > 0 and len(rows) > 0:
            self._inflate_hits(rows, cols)
        else:
            self._dynamic_grid[rows, cols] = 100

        self._publish_combined()

    def _publish_combined(self):
        if self._static_costmap is None or self._map_info is None:
            return

        combined = self._static_costmap.copy()

        if self._dynamic_grid is not None:
            # Zona inflada dinámica (99) solo ocupa celdas libres
            combined[(self._dynamic_grid == 99) & (combined == 0)] = 99
            # Obstáculo dinámico exacto (100) sobreescribe todo
            combined[self._dynamic_grid == 100] = 100

        out = OccupancyGrid()
        out.header.stamp    = self.get_clock().now().to_msg()
        out.header.frame_id = 'map'
        out.info            = self._map_info
        out.data            = combined.flatten().tolist()

        self._costmap_msg = out
        self._pub.publish(out)

    def _inflate_hits(self, hit_rows: np.ndarray, hit_cols: np.ndarray):
        """
        Infla únicamente las celdas hit del LiDAR en _dynamic_grid.
        Mucho más rápido que inflar el grid entero porque solo itera
        sobre N_hits celdas en vez de H×W.

        Para un cilindro de ~30 hits y r=20px:
          30 hits × 1257 offsets = 37k operaciones  ← rápido
          vs 264k celdas × 1257 offsets             ← lento (lo anterior)
        """
        H, W  = self._dynamic_grid.shape
        r_px  = int(math.ceil(self._dyn_r_inf / self._map_res))

        # Precomputar offsets del círculo una sola vez
        dr_v = np.arange(-r_px, r_px + 1)
        dc_v = np.arange(-r_px, r_px + 1)
        DR, DC     = np.meshgrid(dr_v, dc_v, indexing='ij')
        en_circulo = (DR**2 + DC**2) <= r_px**2
        offsets_dr = DR[en_circulo]   # arrays numpy (no tolist, más rápido)
        offsets_dc = DC[en_circulo]

        # Broadcasting: (N_hits, 1) + (1, N_offsets) → (N_hits, N_offsets)
        all_rows = (hit_rows[:, np.newaxis] + offsets_dr).ravel()
        all_cols = (hit_cols[:, np.newaxis] + offsets_dc).ravel()

        # Filtrar celdas dentro del mapa
        in_bounds = (
            (all_rows >= 0) & (all_rows < H) &
            (all_cols >= 0) & (all_cols < W)
        )
        all_rows = all_rows[in_bounds]
        all_cols = all_cols[in_bounds]

        # Marcar zona inflada (99), luego sobrescribir hits exactos con 100
        self._dynamic_grid[all_rows, all_cols] = 99
        self._dynamic_grid[hit_rows, hit_cols] = 100

    def _inflate(self, grid: np.ndarray, resolution: float,
                 radius: float) -> np.ndarray:
        """
        Dilata cada celda ocupada (100) o desconocida (-1) un radio
        en metros, marcando las celdas libres vecinas como 99.

        Algoritmo: desplazamiento de máscara por offset circular.
        Para cada (Δfila, Δcol) dentro del círculo de radio r_px celdas,
        desplaza la máscara fuente y escribe 99 en el destino libre.
        Solo numpy: sin scipy, sin cv2.
        """
        H, W  = grid.shape
        r_px  = int(math.ceil(radius / resolution))

        # Celdas que generan zona de peligro
        fuente  = (grid == 100) | (grid == -1)
        costmap = grid.copy()

        # Precomputar offsets del círculo discreto
        dr_v = np.arange(-r_px, r_px + 1)
        dc_v = np.arange(-r_px, r_px + 1)
        DR, DC     = np.meshgrid(dr_v, dc_v, indexing='ij')
        en_circulo = (DR**2 + DC**2) <= r_px**2
        offsets_dr = DR[en_circulo].tolist()
        offsets_dc = DC[en_circulo].tolist()

        for dr, dc in zip(offsets_dr, offsets_dc):
            r0s = max(0, -dr);  r1s = min(H, H - dr)
            c0s = max(0, -dc);  c1s = min(W, W - dc)
            r0d = r0s + dr;     r1d = r1s + dr
            c0d = c0s + dc;     c1d = c1s + dc

            src  = fuente [r0s:r1s, c0s:c1s]
            dest = costmap[r0d:r1d, c0d:c1d]
            dest[src & (dest == 0)] = 99

        return costmap

    def _world_to_cell_v(self, xs: np.ndarray,
                          ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convierte arrays de coordenadas mundo (xs, ys) a índices de celda
        (cols, rows) del OccupancyGrid, teniendo en cuenta la rotación del mapa.

        Fórmula inversa de la conversión pixel→mundo usada en mcl_node:
          dx = x - origen_x
          dy = y - origen_y
          col = (dx·cos + dy·sin) / res
          row = (−dx·sin + dy·cos) / res
        """
        dx = xs - self._map_ox
        dy = ys - self._map_oy
        cols = (dx * self._map_cos + dy * self._map_sin) / self._map_res
        rows = (-dx * self._map_sin + dy * self._map_cos) / self._map_res
        return cols.astype(int), rows.astype(int)

    def _timer_cb(self):
        if self._costmap_msg is None:
            return
        self._costmap_msg.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(self._costmap_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
