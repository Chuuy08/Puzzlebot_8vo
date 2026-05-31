#!/usr/bin/env python3
"""
rrt_node.py — Paso 3: Planificador global con RRT bidireccional.

Arquitectura:
  /costmap + /goal_pose → rrt_node → /global_path → path_follower → dwa_node

Suscribe:
  /costmap    nav_msgs/OccupancyGrid           mapa con obstáculos inflados
  /mcl_pose   geometry_msgs/PoseWithCovarianceStamped  pose del robot en frame map
  /goal_pose  geometry_msgs/PoseStamped        goal publicado desde RViz o terminal

Publica:
  /global_path  nav_msgs/Path                  trayectoria libre de obstáculos

Algoritmo — RRT Bidireccional (BiRRT):
  Crece dos árboles: árbol A desde el robot, árbol B desde el goal.
  En cada iteración extiende un árbol hacia un punto aleatorio e intenta
  conectar el nuevo nodo con el árbol opuesto. Cuando la conexión es posible,
  extrae el path completo y lo suaviza.
  Converge ~4× más rápido que RRT estándar en mapas con pasillos.

Interpretación del costmap:
   0   → libre (navegable)
  99   → zona inflada (bloqueada para el robot)
  100  → obstáculo (pared o dinámico)
  -1   → desconocido (bloqueado por seguridad)
"""

import math
import random
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

from .utils import quat_to_yaw


_LATCHED = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)
_RELIABLE = QoSProfile(
    depth=5,
    durability=DurabilityPolicy.VOLATILE,
    reliability=ReliabilityPolicy.RELIABLE,
)


class RRTNode(Node):

    def __init__(self):
        super().__init__('rrt_node')

        # ── Parámetros ────────────────────────────────────────────────────
        self.declare_parameter('step_size',       0.20)   # tamaño de paso [m]
        self.declare_parameter('max_iterations',  6000)   # iteraciones máximas
        self.declare_parameter('goal_bias',       0.15)   # prob. de samplear la raíz opuesta
        self.declare_parameter('smooth_path',     True)   # suavizar el path al final

        self._step   = self.get_parameter('step_size').value
        self._max_it = self.get_parameter('max_iterations').value
        self._p_goal = self.get_parameter('goal_bias').value
        self._smooth = self.get_parameter('smooth_path').value

        # ── Estado interno ────────────────────────────────────────────────
        self._grid: np.ndarray | None = None   # snapshot del costmap
        self._map_W   = 0
        self._map_H   = 0
        self._map_res = 0.05
        self._map_ox  = 0.0
        self._map_oy  = 0.0
        self._map_cos = 1.0
        self._map_sin = 0.0

        self._robot_x   = 0.0
        self._robot_y   = 0.0
        self._pose_ready = False
        self._planning   = False       # evitar doble planeación simultánea
        self._grid_lock  = threading.Lock()

        # ── ROS I/O ───────────────────────────────────────────────────────
        self._costmap_sub = self.create_subscription(
            OccupancyGrid, '/costmap', self._costmap_cb, _LATCHED)
        self._pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/mcl_pose', self._pose_cb, _RELIABLE)
        self._goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self._goal_cb, _RELIABLE)
        self._path_pub = self.create_publisher(Path, '/global_path', _RELIABLE)

        self.get_logger().info(
            f'rrt_node listo | step={self._step} m | '
            f'max_iter={self._max_it} | goal_bias={self._p_goal} | '
            f'smooth={self._smooth} | esperando /costmap y /mcl_pose ...')

    # ══════════════════════════════════════════════════════════════════════
    # Callbacks de entrada
    # ══════════════════════════════════════════════════════════════════════

    def _costmap_cb(self, msg: OccupancyGrid):
        with self._grid_lock:
            self._grid    = np.array(msg.data, dtype=np.int8).reshape(
                msg.info.height, msg.info.width)
            self._map_W   = msg.info.width
            self._map_H   = msg.info.height
            self._map_res = msg.info.resolution
            self._map_ox  = msg.info.origin.position.x
            self._map_oy  = msg.info.origin.position.y
            yaw           = quat_to_yaw(msg.info.origin.orientation)
            self._map_cos = math.cos(yaw)
            self._map_sin = math.sin(yaw)

    def _pose_cb(self, msg: PoseWithCovarianceStamped):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        if not self._pose_ready:
            self._pose_ready = True
            self.get_logger().info(
                f'Pose recibida: ({self._robot_x:.2f}, {self._robot_y:.2f}) '
                f'— listo para planificar')

    def _goal_cb(self, msg: PoseStamped):
        # Guardar antes de verificar precondiciones
        if not self._pose_ready:
            self.get_logger().warn(
                'Sin pose del robot. Usa "2D Pose Estimate" en RViz primero.')
            return
        if self._grid is None:
            self.get_logger().warn(
                'Sin costmap disponible. Espera a que /costmap llegue.')
            return
        if self._planning:
            self.get_logger().warn(
                'Ya hay una planeación en curso. Espera a que termine.')
            return

        gx = msg.pose.position.x
        gy = msg.pose.position.y
        start = (self._robot_x, self._robot_y)
        goal  = (gx, gy)

        self.get_logger().info(
            f'Goal recibido: ({gx:.2f}, {gy:.2f}) | '
            f'inicio: ({start[0]:.2f}, {start[1]:.2f})')

        # Planeación en hilo separado: el RRT puede tardar 1-5 s y no debe
        # bloquear el executor principal (TF, heartbeats, etc.)
        self._planning = True
        threading.Thread(
            target=self._plan_thread, args=(start, goal), daemon=True
        ).start()

    # ══════════════════════════════════════════════════════════════════════
    # Hilo de planeación
    # ══════════════════════════════════════════════════════════════════════

    def _plan_thread(self, start: tuple, goal: tuple):
        try:
            # Snapshot del costmap para que los cambios durante la planeación
            # no afecten la consistencia del árbol RRT
            with self._grid_lock:
                grid = self._grid.copy()

            start_a = np.array(start, dtype=np.float64)
            goal_a  = np.array(goal,  dtype=np.float64)

            # Validar que inicio y goal estén en espacio libre
            if not self._pt_free(start_a, grid):
                self.get_logger().error(
                    f'El inicio ({start[0]:.2f},{start[1]:.2f}) está en un '
                    f'obstáculo. Verifica la localización del robot.')
                return
            if not self._pt_free(goal_a, grid):
                self.get_logger().error(
                    f'El goal ({goal[0]:.2f},{goal[1]:.2f}) está en un '
                    f'obstáculo. Elige un punto en espacio libre (blanco en RViz).')
                return

            # Planificar
            path = self._birrt(start_a, goal_a, grid)

            if path is None:
                self.get_logger().warn(
                    f'BiRRT falló en {self._max_it} iteraciones. '
                    f'Intenta un goal más cercano o revisa que haya espacio libre.')
                return

            # Suavizar
            if self._smooth:
                path = self._smooth_path(path, grid)

            self._publish_path(path)

        finally:
            self._planning = False

    # ══════════════════════════════════════════════════════════════════════
    # BiRRT
    # ══════════════════════════════════════════════════════════════════════

    def _birrt(self, start: np.ndarray, goal: np.ndarray,
               grid: np.ndarray):
        """
        RRT Bidireccional.

        Mantiene dos árboles: A (desde start) y B (desde goal).
        En cada iteración:
          1. Extiende árbol A hacia un punto aleatorio.
          2. Intenta conectar árbol B al nuevo nodo.
          3. Si la conexión es libre → path encontrado.
          4. Intercambia A y B para la siguiente iteración.

        Retorna lista de np.array([x, y]) desde start hasta goal,
        o None si no encontró path en max_iterations.
        """
        nodes_a = [start.copy()];  par_a = [-1]
        nodes_b = [goal.copy()];   par_b = [-1]
        a_es_start = True   # para saber cómo ordenar el path al final

        for it in range(self._max_it):
            # ── 1. Muestra aleatoria ──────────────────────────────────────
            # Con probabilidad goal_bias se samplea la raíz del árbol opuesto
            # (= forzar progreso hacia el objetivo)
            if random.random() < self._p_goal:
                q_rand = nodes_b[0].copy()
            else:
                q_rand = self._random_free(grid)
                if q_rand is None:
                    continue

            # ── 2. Extender árbol A ───────────────────────────────────────
            na      = np.array(nodes_a)
            near_i  = int(np.argmin(np.linalg.norm(na - q_rand, axis=1)))
            q_near  = nodes_a[near_i]
            q_new   = self._steer(q_near, q_rand)

            # Verificar que q_new y el segmento q_near→q_new son libres
            if not (self._pt_free(q_new, grid) and
                    self._seg_free(q_near, q_new, grid)):
                # Intercambiar y continuar
                nodes_a, nodes_b = nodes_b, nodes_a
                par_a,   par_b   = par_b,   par_a
                a_es_start       = not a_es_start
                continue

            nodes_a.append(q_new)
            par_a.append(near_i)
            new_i = len(nodes_a) - 1

            # ── 3. Intentar conectar árbol B a q_new ─────────────────────
            nb       = np.array(nodes_b)
            near_bi  = int(np.argmin(np.linalg.norm(nb - q_new, axis=1)))
            q_near_b = nodes_b[near_bi]

            if self._seg_free(q_near_b, q_new, grid):
                # ¡Conexión! Extraer path
                path_a = self._trace(nodes_a, par_a, new_i)    # raíz_A → q_new
                path_b = self._trace(nodes_b, par_b, near_bi)  # raíz_B → q_near_b

                # Ordenar: start → goal
                if a_es_start:
                    # path_a: start→q_new | path_b invertido: q_near_b→goal
                    full = path_a + path_b[::-1]
                else:
                    # path_a: goal→q_new | path_b invertido: q_near_b→start
                    full = path_b + path_a[::-1]   # start→q_near_b + q_new→goal

                self.get_logger().info(
                    f'BiRRT conectado en iteración {it+1} | '
                    f'nodos A={len(nodes_a)} B={len(nodes_b)} | '
                    f'waypoints={len(full)}')
                return full

            # ── 4. Intercambiar árboles ───────────────────────────────────
            nodes_a, nodes_b = nodes_b, nodes_a
            par_a,   par_b   = par_b,   par_a
            a_es_start       = not a_es_start

        return None

    # ══════════════════════════════════════════════════════════════════════
    # Helpers de planeación
    # ══════════════════════════════════════════════════════════════════════

    def _steer(self, from_pt: np.ndarray, to_pt: np.ndarray) -> np.ndarray:
        """Avanza desde from_pt hacia to_pt un máximo de step_size metros."""
        d    = to_pt - from_pt
        dist = float(np.linalg.norm(d))
        if dist <= self._step:
            return to_pt.copy()
        return from_pt + (d / dist) * self._step

    def _pt_free(self, pt: np.ndarray, grid: np.ndarray) -> bool:
        """True si el punto cae en una celda libre (valor 0) del costmap."""
        r, c = self._w2c(float(pt[0]), float(pt[1]))
        if r < 0 or r >= self._map_H or c < 0 or c >= self._map_W:
            return False
        return int(grid[r, c]) == 0

    def _seg_free(self, p1: np.ndarray, p2: np.ndarray,
                  grid: np.ndarray) -> bool:
        """
        True si el segmento p1→p2 está libre de obstáculos.
        Muestrea a intervalos de map_res metros para no perder paredes finas.
        """
        dist = float(np.linalg.norm(p2 - p1))
        if dist < 1e-6:
            return True
        n = max(2, int(math.ceil(dist / self._map_res)))
        for k in range(n + 1):
            t  = k / n
            pt = p1 + t * (p2 - p1)
            if not self._pt_free(pt, grid):
                return False
        return True

    def _random_free(self, grid: np.ndarray):
        """Muestrea un punto aleatorio en espacio libre del mapa."""
        for _ in range(200):
            row = random.randint(0, self._map_H - 1)
            col = random.randint(0, self._map_W - 1)
            if int(grid[row, col]) == 0:
                x, y = self._c2w(row, col)
                return np.array([x, y], dtype=np.float64)
        return None

    def _trace(self, nodes: list, parents: list, idx: int) -> list:
        """Extrae la trayectoria desde la raíz hasta el nodo 'idx'."""
        path = []
        i = idx
        while i != -1:
            path.append(nodes[i].copy())
            i = parents[i]
        return path[::-1]

    def _smooth_path(self, path: list, grid: np.ndarray) -> list:
        """
        Suavizado por atajos (greedy shortcutting):
        intenta conectar directamente waypoints no adyacentes,
        eliminando rodeos innecesarios del árbol RRT.
        """
        if len(path) <= 2:
            return path

        smooth = [path[0]]
        i = 0
        while i < len(path) - 1:
            # Busca el waypoint más lejano alcanzable directamente desde path[i]
            j = len(path) - 1
            while j > i + 1:
                if self._seg_free(path[i], path[j], grid):
                    break
                j -= 1
            smooth.append(path[j])
            i = j

        self.get_logger().info(
            f'Suavizado: {len(path)} → {len(smooth)} waypoints '
            f'({100*(1-len(smooth)/len(path)):.0f}% reducción)')
        return smooth

    # ══════════════════════════════════════════════════════════════════════
    # Conversión de coordenadas mundo ↔ celda del mapa
    # ══════════════════════════════════════════════════════════════════════

    def _w2c(self, x: float, y: float) -> tuple[int, int]:
        """Coordenadas mundo (x,y) → índices de celda (row, col)."""
        dx  = x - self._map_ox
        dy  = y - self._map_oy
        col = int((dx * self._map_cos + dy * self._map_sin) / self._map_res)
        row = int((-dx * self._map_sin + dy * self._map_cos) / self._map_res)
        return row, col

    def _c2w(self, row: int, col: int) -> tuple[float, float]:
        """Índices de celda (row, col) → coordenadas mundo (x, y)."""
        x = self._map_ox + col * self._map_res * self._map_cos \
                         - row * self._map_res * self._map_sin
        y = self._map_oy + col * self._map_res * self._map_sin \
                         + row * self._map_res * self._map_cos
        return x, y

    # ══════════════════════════════════════════════════════════════════════
    # Publicar path
    # ══════════════════════════════════════════════════════════════════════

    def _publish_path(self, path: list):
        msg = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        for pt in path:
            ps = PoseStamped()
            ps.header           = msg.header
            ps.pose.position.x  = float(pt[0])
            ps.pose.position.y  = float(pt[1])
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)

        self._path_pub.publish(msg)


# ══════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = RRTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
