#!/usr/bin/env python3
"""
mission_manager_node.py — Orquestador de misión autónoma del PuzzleBot.

Secuencia: localización (MCL) → RODILLOS → RACK1..3 → fin de misión.

Coordina, vía los topics ya existentes en el sistema:
  - Localización:  /mcl_wandering        (puzzlebot_localisation · active_localization_node)
  - Navegación:    /goal_pose, /waypoint_reached, /cancel_navigation
                   (puzzlebot_navigation · rrt_node + path_follower_node)
  - Visión:        /pallet_detected, /pallet_has_qr, /pallet_qr_content,
                   /alineation/booleano   (puzzlebot_vision · align_and_approach / tracking)
  - Montacargas:   /deteccion_pallet      (puzzlebot_fpga_controller · fpga_controller_node,
                   ya orquesta solo la secuencia de carga/descarga — el orquestador
                   NO necesita publicar ningún comando de "liberación")

NOTA — hueco de sincronización con la FPGA (sin resolver todavía):
  fpga_controller_node no publica NADA hacia afuera (toda su comunicación es
  por SPI). Eso significa que no hay forma de saber, desde ROS, el instante
  exacto en que el montacargas terminó de (a) cargar el pallet tras la
  alineación, o (b) depositarlo en el punto de entrega. Mientras esa señal
  no exista, este nodo usa una espera calibrada (`fpga_settle_time_s`) como
  mitigación — ajústala según los tiempos reales de la FSM de la FPGA
  (ver T_ROD_*/T_RACK_* en fpga_controller_node.py) o reemplázala si agregan
  un publisher de "secuencia completa" del lado de la Jetson.
"""

import math
from enum import Enum, auto

import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String, Bool, Empty
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from rcl_interfaces.srv import GetParameters, SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType


def _quat_to_yaw(w: float, z: float) -> float:
    return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)


def _yaw_to_quat_wz(yaw: float) -> tuple[float, float]:
    return math.cos(yaw / 2.0), math.sin(yaw / 2.0)


def _wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class MissionState(Enum):
    WAITING_LOCALIZATION = auto()    # esperando /mcl_wandering: True -> False

    NAV_TO_AREA_GENERAL = auto()     # navegando al waypoint 'general' del área activa
    ALIGN_TO_YAW        = auto()     # girando en el lugar hacia el yaw grabado del
                                     # waypoint (la navegación no respeta orientación
                                     # final — ver _start_align_to_yaw); se reutiliza
                                     # tras llegar a 'general' y a 'p{N}'
    SWEEP_TURN_TO_STOP  = auto()     # girando hacia la siguiente parada del barrido
    SWEEP_SAMPLING      = auto()     # detenido, muestreando detección de pallet/QR
    SWEEP_EMPTY         = auto()     # ninguna parada tuvo pallet+QR -> decidir siguiente paso

    # NAV_TO_PALLET puede pasar primero por DEFLATE_COSTMAP/INFLATE_COSTMAP
    # (mecanismo genérico, ver _start_costmap_reduce/_restore más abajo):
    # p{N} a veces cae DENTRO del costmap inflado (pegado a un rack) y
    # rrt_node no encuentra ruta — se reduce solo inflation_radius/
    # dynamic_inflation_radius (NO robot_radius, no es pasillo angosto).
    NAV_TO_PALLET     = auto()       # navegando al waypoint p{N} del pallet elegido
    SEND_DETECCION    = auto()       # publicar /deteccion_pallet (solo tras llegar al pallet)
    WAITING_ALIGNMENT = auto()       # esperando /alineation/booleano == True
    WAITING_LOAD      = auto()       # espera calibrada: FPGA terminando de cargar

    # Secuencia de entrega: la zona de clientes cae DENTRO del costmap inflado
    # (está pegada a una pared, en un pasillo angosto) y rrt_node nunca
    # encuentra ruta hacia ella, así que primero se llega a un punto de
    # acceso fijo fuera de esa zona, se reducen EN CALIENTE (servicio
    # set_parameters) inflation_radius/dynamic_inflation_radius de
    # costmap_node Y robot_radius de dwa_node — ver _start_costmap_reduce —,
    # se entra a depositar, y al volver al punto de acceso se restauran los
    # tres a su valor normal con _start_costmap_restore antes de seguir.
    #
    # DEFLATE_COSTMAP/INFLATE_COSTMAP son GENÉRICOS: el mismo mecanismo
    # también lo usa NAV_TO_PALLET (con otros valores objetivo y sin tocar
    # robot_radius — ver _start_costmap_reduce/_poll_costmap_inflation).
    NAV_TO_DELIVERY_ACCESO = auto() # navegando a delivery.acceso (punto fijo, fuera de la zona inflada)
    DEFLATE_COSTMAP        = auto() # reduciendo inflation_radius/dynamic_inflation_radius (y a veces robot_radius) vía set_parameters
    NAV_TO_DELIVERY_ALIGN  = auto() # navegando a delivery.accesoc{N} — punto de pre-alineación
                                    # propio del cliente N: deja al robot ya orientado para entrar
                                    # en línea recta a delivery[clienteN] (la navegación no respeta
                                    # la orientación final del goal, igual que 'general'/'p{N}' —
                                    # ver ALIGN_TO_YAW; aquí se resuelve con un waypoint intermedio
                                    # en vez de un giro porque además hay que recorrer distancia)
    NAV_TO_DELIVERY        = auto() # navegando al waypoint delivery[cliente], ya alineado desde accesoc{N}
    WAITING_UNLOAD         = auto() # espera calibrada: FPGA terminando de depositar
    REVERSE_TO_ACCESO      = auto() # retrocediendo en línea recta a delivery.acceso (sin girar:
                                    # el montacargas queda justo enfrente y no hay espacio para voltearse)
    INFLATE_COSTMAP        = auto() # restaurando inflation_radius/dynamic_inflation_radius (y robot_radius si aplica) a sus valores normales

    MISSION_COMPLETE = auto()
    ABORTED          = auto()


_RELIABLE = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)

# Áreas de recolección, en el orden en que la misión las visita.
# 'rodillos' siempre tiene pallet (es la primera tarea conocida de antemano);
# rack1..rack3 se recorren hasta encontrar el primer pallet con QR.
_PICKUP_AREAS = ['rodillos', 'rack1', 'rack2', 'rack3']
_PALLETS_PER_AREA = {'rodillos': 4, 'rack1': 3, 'rack2': 3, 'rack3': 3}


class MissionManagerNode(Node):

    def __init__(self):
        super().__init__('mission_manager_node')

        self.declare_parameter('waypoints_yaml_path', '')
        self.declare_parameter('sweep_range_deg', 60.0)
        self.declare_parameter('sweep_angular_speed', 0.3)
        self.declare_parameter('sweep_align_threshold_deg', 3.0)
        self.declare_parameter('sweep_settle_time_s', 1.0)
        self.declare_parameter('sweep_samples_per_stop', 5)
        self.declare_parameter('nav_timeout_s', 90.0)
        self.declare_parameter('fpga_settle_time_s', 5.0)
        self.declare_parameter('delivery_inflation_radius', 0.07)
        # robot_radius es el radio FÍSICO del robot + margen mínimo — a
        # diferencia de delivery_inflation_radius (que es solo margen de
        # planeación), bajarlo de más sí puede causar choques reales. Debe
        # quedar apenas por debajo del robot_radius normal de dwa_node
        # (0.18 en sim / 0.13 en el robot real — ver navigation_*_launch.py),
        # nunca por debajo del radio físico real del PuzzleBot.
        self.declare_parameter('delivery_robot_radius', 0.5)
        self.declare_parameter('costmap_settle_time_s', 1.0)
        # El pasillo de entrega es angosto y tiene el montacargas justo
        # enfrente del robot al llegar — no hay espacio para girar 180°, así
        # que el regreso a delivery.acceso se hace EN REVERSA, en línea recta
        # (ver _start_reverse_to_acceso). reverse_speed va por debajo de
        # linear_speed de path_follower porque no hay sensado de obstáculos
        # hacia atrás (el LiDAR ve hacia adelante).
        self.declare_parameter('reverse_speed', 0.10)            # [m/s], magnitud (se aplica negativo)
        self.declare_parameter('reverse_goal_tolerance', 0.15)   # [m]
        self.declare_parameter('reverse_heading_kp', 1.0)        # ganancia P para mantener el rumbo inicial
        # Freno de proximidad trasero durante la reversa: el robot no "ve"
        # literalmente hacia atrás (no hay cámara), pero el LiDAR (sllidar,
        # RPLidar) SÍ es de 360° — los mismos /scan que usa dwa_node también
        # traen los rebotes de detrás. scan_topic/laser_angle_offset DEBEN
        # coincidir con los que usa navigation_*_launch.py (/scan + 0.0 en
        # sim, /scan_fixed + π en el robot real — ahí el cable del RPLidar
        # apunta hacia atrás) para interpretar los ángulos correctamente.
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('laser_angle_offset', 0.0)
        self.declare_parameter('reverse_min_clearance', 0.15)    # [m] frena si algo se acerca más que esto
        self.declare_parameter('reverse_rear_arc_deg', 100.0)    # semiángulo del arco trasero vigilado
        # Igual que delivery_*: el waypoint de alineación a un pallet (p{N})
        # puede caer DENTRO del costmap inflado si está pegado a un rack,
        # y rrt_node nunca encuentra ruta — se reduce solo inflation_radius/
        # dynamic_inflation_radius (planeación global) antes de acercarse;
        # a diferencia de la entrega, aquí NO hace falta tocar robot_radius
        # de dwa_node (no es un pasillo angosto, solo el punto cae en zona
        # inflada — basta con que rrt_node encuentre ruta).
        self.declare_parameter('pallet_inflation_radius', 0.10)

        self._sweep_range   = math.radians(self.get_parameter('sweep_range_deg').value)
        self._sweep_speed   = self.get_parameter('sweep_angular_speed').value
        self._sweep_align_th = math.radians(self.get_parameter('sweep_align_threshold_deg').value)
        self._sweep_settle  = self.get_parameter('sweep_settle_time_s').value
        self._sweep_n_samples = int(self.get_parameter('sweep_samples_per_stop').value)
        self._nav_timeout   = self.get_parameter('nav_timeout_s').value
        self._fpga_settle   = self.get_parameter('fpga_settle_time_s').value
        self._delivery_inflation = self.get_parameter('delivery_inflation_radius').value
        self._delivery_robot_radius = self.get_parameter('delivery_robot_radius').value
        self._costmap_settle_time = self.get_parameter('costmap_settle_time_s').value
        self._reverse_speed = self.get_parameter('reverse_speed').value
        self._reverse_tolerance = self.get_parameter('reverse_goal_tolerance').value
        self._reverse_heading_kp = self.get_parameter('reverse_heading_kp').value
        self._scan_topic = self.get_parameter('scan_topic').value
        self._laser_offset = self.get_parameter('laser_angle_offset').value
        self._reverse_min_clearance = self.get_parameter('reverse_min_clearance').value
        self._reverse_rear_halfarc = math.radians(self.get_parameter('reverse_rear_arc_deg').value)
        self._pallet_inflation = self.get_parameter('pallet_inflation_radius').value

        yaml_path = self.get_parameter('waypoints_yaml_path').value
        if not yaml_path:
            raise RuntimeError(
                "Parámetro 'waypoints_yaml_path' vacío — pasa la ruta al "
                "waypoints.yaml (p.ej. vía el launch con get_package_share_directory).")
        self._waypoints = self._load_waypoints_yaml(yaml_path)

        # ---- Estado de la FSM ----
        self._state = MissionState.WAITING_LOCALIZATION
        self._area_idx = 0                  # índice en _PICKUP_AREAS
        self._cliente_actual: str | None = None
        self._pallet_idx_objetivo: int | None = None

        self._goal_active = False
        self._goal_start_time = None          # rclpy.time.Time — para detectar timeout
        self._nav_pending_target: tuple[str, str] | None = None  # (area, punto) en vuelo

        self._rx = self._ry = 0.0             # posición actual (de /mcl_pose)
        self._rth = 0.0                       # yaw actual (de /mcl_pose)
        self._pose_ready = False

        self._sweep_stops: list[float] = []
        self._sweep_idx = 0
        self._sweep_phase_start = None         # Time: inicio del asentamiento de la parada actual
        self._sweep_samples: list[tuple[bool, bool, str | None]] = []

        self._align_target_yaw: float | None = None
        self._align_on_done = None              # callable() invocado al terminar el giro

        self._fpga_wait_start = None            # Time: inicio de la espera calibrada de FPGA

        # ---- Maniobra de reversa (regreso a delivery.acceso) ----
        # ver _start_reverse_to_acceso/_do_reverse_to_acceso
        self._reverse_heading0: float | None = None   # rumbo a mantener (capturado al iniciar)
        self._reverse_target: dict | None = None      # {'x':.., 'y':..} de delivery.acceso
        self._reverse_start_time = None               # Time: para reverse_timeout (reusa nav_timeout_s)
        self._reverse_on_done = None
        # Freno de proximidad trasero (ver _cb_scan / _do_reverse_to_acceso):
        # None = todavía no llega ningún /scan → se trata como "frenar"
        # (seguro por defecto, igual que dwa_node sin datos de sensor).
        self._rear_clearance: float | None = None
        self._reverse_braking = False

        # ---- Control dinámico de margen del costmap/dwa (zonas estrechas) ----
        # ver _start_costmap_reduce/_start_costmap_restore/_poll_costmap_inflation
        # Mecanismo GENÉRICO reusado en dos situaciones (cada una con sus
        # propios valores objetivo — ver delivery_*/pallet_* arriba):
        #   - entrega: el punto cae en un pasillo angosto pegado a la pared
        #     → hace falta reducir LOS TRES (si no, dwa_node sigue frenando
        #     y oscilando aunque rrt_node ya encuentre ruta)
        #   - acercarse a un pallet (p{N}): el punto puede caer DENTRO del
        #     costmap inflado (pegado al rack) y rrt_node no encuentra ruta
        #     → basta reducir costmap (planeación global); robot_radius se
        #     deja en su valor normal (ver _costmap_keep_robot_radius)
        self._normal_inflation: float | None = None           # se consultan una sola vez (lazy)
        self._normal_dynamic_inflation: float | None = None
        self._normal_robot_radius: float | None = None
        self._costmap_phase: str | None = None
        # 'query_costmap' | 'query_dwa' | 'set_costmap' | 'set_dwa' | 'settling'
        self._costmap_future = None
        self._costmap_settle_start = None
        self._costmap_target_value: float | None = None
        self._costmap_target_dynamic: float | None = None
        self._costmap_target_robot_radius: float | None = None
        self._costmap_keep_robot_radius = False  # True → robot_radius objetivo = su valor normal (no-op)
        self._costmap_on_done = None

        # ---- Suscripciones a estado del sistema ----
        self.create_subscription(Bool, '/mcl_wandering', self._cb_mcl_wandering, 10)
        self.create_subscription(Empty, '/waypoint_reached', self._cb_waypoint_reached, _RELIABLE)
        from geometry_msgs.msg import PoseWithCovarianceStamped
        self.create_subscription(
            PoseWithCovarianceStamped, '/mcl_pose', self._cb_mcl_pose, _RELIABLE)
        # Mismo /scan de 360° que usa dwa_node — aquí solo para el freno de
        # proximidad trasero durante _do_reverse_to_acceso (ver _cb_scan).
        self.create_subscription(LaserScan, self._scan_topic, self._cb_scan, _RELIABLE)

        # ---- Suscripciones a visión (topics agregados a align_and_approach/tracking) ----
        self.create_subscription(Bool, '/pallet_detected', self._cb_pallet_detected, _RELIABLE)
        self.create_subscription(Bool, '/pallet_has_qr', self._cb_pallet_has_qr, _RELIABLE)
        self.create_subscription(String, '/pallet_qr_content', self._cb_pallet_qr_content, _RELIABLE)
        self.create_subscription(Bool, '/alineation/booleano', self._cb_alineacion, _RELIABLE)

        # último valor recibido de cada topic de visión (para muestreo por parada)
        self._last_pallet_detected: bool | None = None
        self._last_pallet_has_qr: bool | None = None
        self._last_qr_content: str | None = None

        # ---- Publishers hacia el resto del sistema ----
        self._goal_pub      = self.create_publisher(PoseStamped, '/goal_pose', _RELIABLE)
        self._cancel_pub    = self.create_publisher(Empty, '/cancel_navigation', 10)
        self._deteccion_pub = self.create_publisher(String, '/deteccion_pallet', _RELIABLE)
        # Anuncia el inicio de cada barrido (nombre del área) — lo usa
        # vision_faker para reiniciar su estado simulado automáticamente al
        # cambiar de área (ver _start_sweep / VisionFaker._cb_sweep_area);
        # también sirve como gancho genérico para cualquier observador
        # externo (GUI, logging) que quiera saber en qué área está parado.
        self._sweep_area_pub = self.create_publisher(String, '/mission_sweep_area', _RELIABLE)
        self._cmd_vel_pub   = self.create_publisher(Twist, '/cmd_vel', _RELIABLE)
        # Bandera para que dwa_node ceda /cmd_vel mientras este nodo gira durante
        # el barrido (replica el patrón de /mcl_wandering). REQUIERE modificar
        # dwa_node para que la respete — ver advertencia pendiente.
        self._ext_cmd_pub = self.create_publisher(Bool, '/external_cmd_vel_active', 10)

        # Clientes de servicio hacia costmap_node y dwa_node — para
        # desinflar/inflar la zona de entrega y reducir/restaurar el margen
        # de evitación local en caliente (ver _poll_costmap_inflation).
        # Requiere que ambos nodos tengan add_on_set_parameters_callback
        # (agregado junto con esta función).
        self._costmap_get_client = self.create_client(GetParameters, '/costmap_node/get_parameters')
        self._costmap_set_client = self.create_client(SetParameters, '/costmap_node/set_parameters')
        self._dwa_get_client = self.create_client(GetParameters, '/dwa_node/get_parameters')
        self._dwa_set_client = self.create_client(SetParameters, '/dwa_node/set_parameters')

        self.create_timer(0.1, self._loop)

        self.get_logger().info(
            f'mission_manager listo | áreas={_PICKUP_AREAS} | '
            f'esperando convergencia de MCL (/mcl_wandering)...')

    # ───────────────────────── Carga de configuración ─────────────────────────

    def _load_waypoints_yaml(self, path: str) -> dict:
        """Carga el .yaml de waypoints y valida que existan todas las áreas
        y sub-puntos que la misión necesita (delivery[cliente1..3],
        cada área de _PICKUP_AREAS con 'general' y p1..pN según
        _PALLETS_PER_AREA, cada uno con position{x,y} y orientation{w})."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if 'delivery' not in data:
            raise RuntimeError(f"waypoints.yaml: falta la clave 'delivery' en {path}")
        for cliente in ('cliente1', 'cliente2', 'cliente3'):
            self._validar_punto(data['delivery'], cliente, f'delivery.{cliente}')
            accesoc = cliente.replace('cliente', 'accesoc')
            self._validar_punto(data['delivery'], accesoc, f'delivery.{accesoc}')
        self._validar_punto(data['delivery'], 'acceso', 'delivery.acceso')

        for area in _PICKUP_AREAS:
            if area not in data:
                raise RuntimeError(f"waypoints.yaml: falta el área '{area}' en {path}")
            self._validar_punto(data[area], 'general', f'{area}.general')
            for i in range(1, _PALLETS_PER_AREA[area] + 1):
                self._validar_punto(data[area], f'p{i}', f'{area}.p{i}')

        return data

    @staticmethod
    def _validar_punto(area_dict: dict, clave: str, ruta: str):
        if clave not in area_dict:
            raise RuntimeError(f"waypoints.yaml: falta el punto '{ruta}'")
        punto = area_dict[clave]
        if 'position' not in punto or 'x' not in punto['position'] or 'y' not in punto['position']:
            raise RuntimeError(f"waypoints.yaml: '{ruta}.position' incompleto (requiere x, y)")
        if 'orientation' not in punto or 'w' not in punto['orientation']:
            raise RuntimeError(f"waypoints.yaml: '{ruta}.orientation' incompleto (requiere w)")

    def _waypoint_to_pose(self, area: str, punto: str) -> PoseStamped:
        """Construye un PoseStamped (frame_id='map') a partir de
        self._waypoints[area][punto]. 'orientation' solo trae quaternion
        plano de yaw (w, y opcionalmente z — si falta, se asume z=0)."""
        wp = self._waypoints[area][punto]
        pos = wp['position']
        ori = wp['orientation']

        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(pos['x'])
        msg.pose.position.y = float(pos['y'])
        msg.pose.orientation.w = float(ori['w'])
        msg.pose.orientation.z = float(ori.get('z', 0.0))
        return msg

    def _yaw_de_waypoint(self, area: str, punto: str) -> float:
        ori = self._waypoints[area][punto]['orientation']
        return _quat_to_yaw(float(ori['w']), float(ori.get('z', 0.0)))

    # ───────────────────────── Navegación ─────────────────────────

    def _elapsed_s(self, start) -> float:
        """Segundos transcurridos desde 'start' (rclpy.time.Time). Mismo
        patrón que usa path_follower_node para medir tiempos sin Duration."""
        return (self.get_clock().now() - start).nanoseconds / 1e9

    def _send_nav_goal(self, pose: PoseStamped):
        """Publica el goal en /goal_pose, marca self._goal_active = True
        y arranca el cronómetro de timeout (nav_timeout_s). Si
        /waypoint_reached no llega a tiempo, _loop() debe abortar la
        misión (path_follower ya re-planifica solo ante atascos — un
        timeout aquí indica algo más grave, p.ej. goal inalcanzable)."""
        self._goal_pub.publish(pose)
        self._goal_active = True
        self._goal_start_time = self.get_clock().now()
        self.get_logger().info(
            f'Goal enviado → ({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f})')

    def _cb_waypoint_reached(self, _msg: Empty):
        """Confirma la llegada al goal actual. CRÍTICO: solo procesar si
        self._goal_active es True (descarta ecos viejos), y apagar la
        bandera de inmediato — para que no quede ninguna navegación
        "fantasma" que dispare al fpga_controller_node fuera de tiempo
        (ver nota de carrera de /waypoint_reached en el diseño)."""
        if not self._goal_active:
            return
        self._goal_active = False
        self._goal_start_time = None
        self.get_logger().info(f'Waypoint alcanzado | estado={self._state.name}')

    def _cb_mcl_pose(self, msg):
        self._rx = msg.pose.pose.position.x
        self._ry = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._rth = _quat_to_yaw(q.w, q.z)
        self._pose_ready = True

    def _cb_scan(self, msg: LaserScan):
        """Guarda en self._rear_clearance la distancia mínima detectada en
        un arco ancho centrado en 'atrás' del robot (±reverse_rear_arc_deg,
        corregido por laser_angle_offset — el LiDAR es de 360°, así que SÍ
        cubre la parte trasera aunque el robot no tenga cámara hacia allá).
        Se usa SOLO como freno de proximidad durante _do_reverse_to_acceso;
        se mantiene siempre actualizado (no solo durante la reversa) para
        que el primer ciclo de la maniobra ya tenga un dato fresco."""
        amin = msg.angle_min
        ainc = msg.angle_increment
        rmin = msg.range_min
        rmax = msg.range_max
        best = float('inf')
        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r) or r < rmin or r > rmax:
                continue
            a = _wrap_angle(amin + i * ainc + self._laser_offset - math.pi)
            if abs(a) < self._reverse_rear_halfarc and r < best:
                best = r
        self._rear_clearance = best if best != float('inf') else None

    # ───────────────────────── Localización (Fase 0) ─────────────────────────

    def _cb_mcl_wandering(self, msg: Bool):
        """active_localization_node publica False cuando MCL convergió de
        forma sostenida (ver su log: 'envía un 2D Goal Pose para navegar').
        En la transición True/desconocido -> False con el robot todavía en
        WAITING_LOCALIZATION, arranca la misión navegando al primer área."""
        if not msg.data and self._state == MissionState.WAITING_LOCALIZATION:
            self.get_logger().info('MCL convergido → iniciando misión (RODILLOS)')
            self._area_idx = 0
            self._state = MissionState.NAV_TO_AREA_GENERAL

    # ───────────────────────── Barrido rotacional (pasos discretos) ──────────

    def _start_sweep(self, area: str):
        """Calcula las N paradas del barrido (N = _PALLETS_PER_AREA[area]).

        'general' está grabado mirando ya hacia la primera posición — la más
        a la izquierda, pallet 1 — así que la parada 0 coincide con esa misma
        orientación (el robot ya llegó alineado a ella vía ALIGN_TO_YAW) y
        las siguientes avanzan girando hacia la DERECHA (yaw decreciente, en
        pasos de sweep_range/(N-1)) hasta cubrir sweep_range_deg en total.
        Reinicia los índices de barrido y pasa a SWEEP_TURN_TO_STOP.

        Cada parada i (0-indexed) corresponde al waypoint p{i+1} — el
        barrido va de izquierda a derecha igual que la numeración."""
        n = _PALLETS_PER_AREA[area]
        start = self._yaw_de_waypoint(area, 'general')
        if n > 1:
            step = self._sweep_range / (n - 1)
            self._sweep_stops = [_wrap_angle(start - i * step) for i in range(n)]
        else:
            self._sweep_stops = [start]
        self._sweep_idx = 0
        self._sweep_samples = []
        self._sweep_phase_start = None
        self.get_logger().info(
            f'Barrido iniciado en {area} | {n} paradas | rango={math.degrees(self._sweep_range):.0f}°')
        self._sweep_area_pub.publish(String(data=area))
        self._publish_external_cmd_active(True)

    def _stop_sweep(self):
        """Detiene cualquier giro de barrido en curso (Twist() vacío),
        cede el control de /cmd_vel de vuelta al stack de navegación, y
        limpia el estado interno del barrido."""
        self._cmd_vel_pub.publish(Twist())
        self._publish_external_cmd_active(False)
        self._sweep_stops = []
        self._sweep_idx = 0
        self._sweep_samples = []
        self._sweep_phase_start = None

    def _publish_external_cmd_active(self, active: bool):
        """Avisa que este nodo (no el stack de navegación) controla /cmd_vel.
        REQUIERE que dwa_node se suscriba a este topic y ceda el control
        igual que ya hace con /mcl_wandering — ese cambio todavía está
        pendiente (ver advertencias)."""
        self._ext_cmd_pub.publish(Bool(data=active))

    def _do_sweep_turn(self):
        """Gira en el lugar hacia self._sweep_stops[self._sweep_idx] con
        control proporcional simple (igual patrón que _do_align en
        path_follower_node). Al quedar dentro de sweep_align_threshold_deg,
        detiene el giro, reinicia el buffer de muestras y pasa a
        SWEEP_SAMPLING tras sweep_settle_time_s de asentamiento."""
        target = self._sweep_stops[self._sweep_idx]
        err = _wrap_angle(target - self._rth)

        if abs(err) < self._sweep_align_th:
            self._cmd_vel_pub.publish(Twist())
            self._sweep_samples = []
            self._sweep_phase_start = self.get_clock().now()
            self._state = MissionState.SWEEP_SAMPLING
            self.get_logger().info(
                f'Parada {self._sweep_idx + 1}/{len(self._sweep_stops)} alcanzada → asentando...')
            return

        t = Twist()
        t.angular.z = math.copysign(self._sweep_speed, err)
        self._cmd_vel_pub.publish(t)

    # ───────────────────── Giro a yaw grabado (post-navegación) ──────────────

    def _start_align_to_yaw(self, target_yaw: float, on_done):
        """Arranca un giro en el lugar hacia 'target_yaw' (mismo control
        proporcional que _do_sweep_turn) y, al terminar, invoca 'on_done'
        (callable sin argumentos) para que el llamador decida cómo continuar.

        Existe porque el stack de navegación IGNORA la orientación final del
        /goal_pose: rrt_node fuerza orientation.w=1.0 al construir el
        /global_path, y path_follower_node solo compara distancia (x, y) para
        publicar /waypoint_reached — el robot puede llegar a 'general' o a
        'p{N}' mirando hacia cualquier lado. Este paso fuerza que termine
        mirando exactamente hacia el yaw grabado en el .yaml (necesario para
        que el barrido arranque alineado, y para que la cámara quede apuntando
        hacia el pallet al llegar a 'p{N}')."""
        self._align_target_yaw = target_yaw
        self._align_on_done = on_done
        self._state = MissionState.ALIGN_TO_YAW
        self._publish_external_cmd_active(True)

    def _do_align_to_yaw(self):
        err = _wrap_angle(self._align_target_yaw - self._rth)

        if abs(err) < self._sweep_align_th:
            self._cmd_vel_pub.publish(Twist())
            self._publish_external_cmd_active(False)
            on_done = self._align_on_done
            self._align_target_yaw = None
            self._align_on_done = None
            on_done()
            return

        t = Twist()
        t.angular.z = math.copysign(self._sweep_speed, err)
        self._cmd_vel_pub.publish(t)

    # ───────────────── Reversa de salida (zona de entrega) ───────────────────

    def _start_reverse_to_acceso(self, on_done):
        """Arranca el regreso a delivery.acceso EN REVERSA, en línea recta,
        sin girar 180° — el pasillo de entrega es angosto y el robot llega
        con el montacargas justo enfrente, así que no hay espacio para darse
        la vuelta; debe retroceder por el mismo camino por el que entró.

        Control simple: mantiene el rumbo que tenía al terminar de entregar
        (el mismo con el que entró al pasillo) con una corrección
        proporcional, avanza con velocidad lineal NEGATIVA constante, y se
        detiene si el LiDAR (360°, ver _cb_scan) detecta algo demasiado
        cerca por detrás — el robot no tiene cámara hacia atrás, pero el
        láser sí "ve" en esa dirección. Termina cuando la distancia a
        delivery.acceso cae bajo reverse_goal_tolerance — igual que
        _ensure_nav_goal pero sin pasar por rrt_node/path_follower (que
        solo saben ir hacia adelante)."""
        self._reverse_heading0 = self._rth
        self._reverse_target = self._waypoints['delivery']['acceso']['position']
        self._reverse_start_time = self.get_clock().now()
        self._reverse_on_done = on_done
        self._reverse_braking = False
        self._state = MissionState.REVERSE_TO_ACCESO
        self._publish_external_cmd_active(True)
        self.get_logger().info(
            f'Retrocediendo a delivery.acceso en línea recta '
            f'(rumbo a mantener={math.degrees(self._reverse_heading0):.1f}°)')

    def _do_reverse_to_acceso(self):
        tx = float(self._reverse_target['x'])
        ty = float(self._reverse_target['y'])
        dist = math.hypot(tx - self._rx, ty - self._ry)

        if dist < self._reverse_tolerance:
            self._cmd_vel_pub.publish(Twist())
            self._publish_external_cmd_active(False)
            on_done = self._reverse_on_done
            self._reverse_heading0 = None
            self._reverse_target = None
            self._reverse_start_time = None
            self._reverse_on_done = None
            self.get_logger().info(f'delivery.acceso alcanzado en reversa | error={dist:.3f} m')
            on_done()
            return

        if self._elapsed_s(self._reverse_start_time) > self._nav_timeout:
            self.get_logger().error(
                f'Timeout retrocediendo a delivery.acceso (faltan {dist:.2f} m) '
                f'→ abortando misión')
            self._cmd_vel_pub.publish(Twist())
            self._publish_external_cmd_active(False)
            self._state = MissionState.ABORTED
            return

        # Freno de proximidad trasero: el robot no tiene cámara hacia atrás,
        # pero el LiDAR es de 360° y sí "ve" en esa dirección (_cb_scan).
        # Solo frena (no esquiva) — esquivar en reversa requeriría replanear,
        # y el timeout de arriba aborta la misión si queda atrapado.
        braking = (self._rear_clearance is None
                   or self._rear_clearance < self._reverse_min_clearance)
        if braking != self._reverse_braking:
            self._reverse_braking = braking
            if braking:
                reason = ('sin datos de scan aún' if self._rear_clearance is None
                          else f'obstáculo a {self._rear_clearance:.2f} m')
                self.get_logger().warn(f'Reversa: freno de proximidad activado ({reason})')
            else:
                self.get_logger().info('Reversa: freno de proximidad liberado, continuando')

        if braking:
            self._cmd_vel_pub.publish(Twist())
            return

        # Corrección proporcional de rumbo (válida igual yendo hacia adelante
        # o en reversa: theta_dot = omega no depende del signo de v).
        err = _wrap_angle(self._reverse_heading0 - self._rth)
        w = max(-self._sweep_speed, min(self._sweep_speed, self._reverse_heading_kp * err))

        t = Twist()
        t.linear.x  = -self._reverse_speed
        t.angular.z = float(w)
        self._cmd_vel_pub.publish(t)

    # ───────────────── Inflado/desinflado dinámico del costmap (entrega) ─────

    def _start_costmap_reduce(self, target_inflation, target_robot_radius, on_done):
        """Arranca la secuencia GENÉRICA para reducir, A LA VEZ, los
        parámetros de otros nodos que controlan qué tan cerca de un
        obstáculo puede ir/planear el robot (ver _poll_costmap_inflation
        para el detalle de fases):
          - costmap_node.inflation_radius / dynamic_inflation_radius (capas
            estática y dinámica — si solo se tocara una, la otra volvería a
            tapar la zona y rrt_node seguiría sin encontrar ruta) → SIEMPRE
            se reducen a target_inflation
          - dwa_node.robot_radius (margen de evitación local) → solo si
            target_robot_radius no es None; si es None se deja en su valor
            normal (caso pallet: el problema es que rrt_node no encuentra
            ruta porque el punto cae en zona inflada, NO que dwa_node oscile
            — tocar robot_radius ahí sería un riesgo innecesario)

        La primera vez que se llama (a CUALQUIERA de los dos escenarios que
        usan este mecanismo — entrega o pallet) también consulta vía
        get_parameters y recuerda los tres valores normales, para
        restaurarlos exactos más adelante con _start_costmap_restore — evita
        duplicarlos como parámetros propios (los de simulación y los del
        robot real son distintos)."""
        if not self._delivery_clearance_services_ready():
            self.get_logger().error(
                'Servicios de parámetros de costmap_node/dwa_node no disponibles '
                '(¿están corriendo?) → abortando misión')
            self._state = MissionState.ABORTED
            return
        self._costmap_on_done = on_done
        self._costmap_target_value = target_inflation
        self._costmap_target_dynamic = target_inflation
        self._costmap_keep_robot_radius = (target_robot_radius is None)
        self._costmap_target_robot_radius = target_robot_radius
        self._state = MissionState.DEFLATE_COSTMAP
        if self._normal_inflation is not None:
            self._begin_set_costmap_params()
        else:
            self._costmap_phase = 'query_costmap'
            self._costmap_future = self._costmap_get_client.call_async(
                GetParameters.Request(names=['inflation_radius', 'dynamic_inflation_radius']))

    def _start_costmap_restore(self, on_done):
        """Restaura los tres parámetros (costmap_node.inflation_radius/
        dynamic_inflation_radius, dwa_node.robot_radius) a los valores
        normales recordados por la primera llamada a _start_costmap_reduce
        (ya consultados, no hace falta volver a pedirlos). Sirve para
        ambos escenarios (entrega y pallet) — el destino es siempre
        'el valor normal', sin importar cuál de los dos lo redujo."""
        if not self._delivery_clearance_services_ready():
            self.get_logger().error(
                'Servicios de parámetros de costmap_node/dwa_node no disponibles '
                '(¿siguen corriendo?) → abortando misión sin restaurarlos')
            self._state = MissionState.ABORTED
            return
        self._costmap_on_done = on_done
        self._costmap_target_value = self._normal_inflation
        self._costmap_target_dynamic = self._normal_dynamic_inflation
        self._costmap_keep_robot_radius = False
        self._costmap_target_robot_radius = self._normal_robot_radius
        self._state = MissionState.INFLATE_COSTMAP
        self._begin_set_costmap_params()

    def _delivery_clearance_services_ready(self) -> bool:
        return (self._costmap_get_client.service_is_ready()
                and self._costmap_set_client.service_is_ready()
                and self._dwa_get_client.service_is_ready()
                and self._dwa_set_client.service_is_ready())

    def _begin_set_costmap_params(self):
        self._costmap_phase = 'set_costmap'
        req = SetParameters.Request()
        req.parameters = [
            Parameter(name='inflation_radius',
                      value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                           double_value=float(self._costmap_target_value))),
            Parameter(name='dynamic_inflation_radius',
                      value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                           double_value=float(self._costmap_target_dynamic))),
        ]
        self._costmap_future = self._costmap_set_client.call_async(req)

    def _begin_set_dwa_radius(self):
        self._costmap_phase = 'set_dwa'
        # Caso pallet (_costmap_keep_robot_radius=True): no tocar robot_radius
        # — se "restaura" a sí mismo (no-op real, pero mantiene la secuencia
        # uniforme y el log informativo de abajo coherente).
        target = (self._normal_robot_radius if self._costmap_keep_robot_radius
                  else self._costmap_target_robot_radius)
        req = SetParameters.Request()
        req.parameters = [Parameter(
            name='robot_radius',
            value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                 double_value=float(target)))]
        self._costmap_future = self._dwa_set_client.call_async(req)

    def _costmap_seq_request_ok(self) -> bool:
        """Revisa el resultado de la petición set_parameters en curso de la
        secuencia; si algún nodo la rechazó, aborta la misión. Devuelve True
        solo si fue exitosa (para que el llamador siga a la siguiente fase)."""
        result = self._costmap_future.result()
        if not result.results or not all(r.successful for r in result.results):
            self.get_logger().error(
                'Un nodo rechazó el cambio de margen de la zona de entrega → abortando misión')
            self._state = MissionState.ABORTED
            return False
        return True

    def _poll_costmap_inflation(self):
        """Avanza la secuencia asíncrona de fases para cambiar, sin bloquear
        el executor, los tres parámetros que controlan qué tan cerca puede
        llegar el robot a las paredes de la zona de entrega:
          'query_costmap' (solo la 1ª vez, get_parameters de costmap_node →
                           recuerda inflation_radius/dynamic_inflation_radius)
            -> 'query_dwa' (solo la 1ª vez, get_parameters de dwa_node →
                            recuerda robot_radius)
            -> 'set_costmap' (set_parameters con ambas inflaciones objetivo)
            -> 'set_dwa'     (set_parameters con robot_radius objetivo)
            -> 'settling'    (costmap_settle_time_s — costmap_node recalcula
                              _static_costmap y republica /costmap antes de seguir)
            -> on_done()
        Cualquier falla de servicio aborta la misión (no tiene sentido seguir
        si no se puede confiar en qué tan cerca de las paredes puede ir el robot)."""
        if self._costmap_phase == 'query_costmap':
            if not self._costmap_future.done():
                return
            values = self._costmap_future.result().values
            self._normal_inflation = float(values[0].double_value)
            self._normal_dynamic_inflation = float(values[1].double_value)
            self._costmap_phase = 'query_dwa'
            self._costmap_future = self._dwa_get_client.call_async(
                GetParameters.Request(names=['robot_radius']))
            return

        if self._costmap_phase == 'query_dwa':
            if not self._costmap_future.done():
                return
            self._normal_robot_radius = float(self._costmap_future.result().values[0].double_value)
            self.get_logger().info(
                f'valores normales → inflation_radius={self._normal_inflation} m | '
                f'dynamic_inflation_radius={self._normal_dynamic_inflation} m | '
                f'robot_radius={self._normal_robot_radius} m')
            self._begin_set_costmap_params()
            return

        if self._costmap_phase == 'set_costmap':
            if not self._costmap_future.done():
                return
            if not self._costmap_seq_request_ok():
                return
            self.get_logger().info(
                f'costmap_node: inflation_radius/dynamic_inflation_radius → '
                f'{self._costmap_target_value} m / {self._costmap_target_dynamic} m')
            self._begin_set_dwa_radius()
            return

        if self._costmap_phase == 'set_dwa':
            if not self._costmap_future.done():
                return
            if not self._costmap_seq_request_ok():
                return
            radius_msg = ('robot_radius sin cambios (caso pallet)'
                          if self._costmap_keep_robot_radius
                          else f'robot_radius → {self._costmap_target_robot_radius} m')
            self.get_logger().info(
                f'dwa_node: {radius_msg} (asentando {self._costmap_settle_time:.1f} s)')
            self._costmap_phase = 'settling'
            self._costmap_settle_start = self.get_clock().now()
            return

        if self._costmap_phase == 'settling':
            if self._elapsed_s(self._costmap_settle_start) < self._costmap_settle_time:
                return
            on_done = self._costmap_on_done
            self._costmap_phase = None
            self._costmap_future = None
            self._costmap_settle_start = None
            self._costmap_target_value = None
            self._costmap_target_dynamic = None
            self._costmap_target_robot_radius = None
            self._costmap_keep_robot_radius = False
            self._costmap_on_done = None
            on_done()

    def _do_sweep_sampling(self):
        """Espera sweep_settle_time_s (para que la imagen deje de moverse),
        luego acumula sweep_samples_per_stop lecturas de
        (/pallet_detected, /pallet_has_qr, /pallet_qr_content) según van
        llegando. Al completar la muestra, evalúa con _evaluate_sweep_stop():
          - si hay pallet+QR consistente → cachear cliente + índice,
            _stop_sweep() y pasar a NAV_TO_PALLET (early exit)
          - si no → avanzar self._sweep_idx; si quedan paradas, volver a
            SWEEP_TURN_TO_STOP; si no, pasar a SWEEP_EMPTY"""
        if self._elapsed_s(self._sweep_phase_start) < self._sweep_settle:
            return

        # Ya asentado: registrar una lectura por ciclo de _loop (10 Hz)
        if self._last_pallet_detected is not None:
            self._sweep_samples.append((
                self._last_pallet_detected,
                bool(self._last_pallet_has_qr),
                self._last_qr_content,
            ))

        if len(self._sweep_samples) < self._sweep_n_samples:
            return

        cliente = self._evaluate_sweep_stop()
        if cliente is not None:
            self._pallet_idx_objetivo = self._sweep_idx + 1
            self._cliente_actual = cliente
            self.get_logger().info(
                f'Pallet con QR encontrado → índice={self._pallet_idx_objetivo} cliente={cliente}')
            self._stop_sweep()
            # El waypoint p{N} puede caer DENTRO del costmap inflado (pegado
            # al rack) y rrt_node no encontrar ruta — se reduce primero
            # inflation_radius/dynamic_inflation_radius (mismo mecanismo
            # genérico que usa la entrega, ver _start_costmap_reduce; aquí
            # con target_robot_radius=None: no se toca dwa_node.robot_radius).
            self._start_costmap_reduce(self._pallet_inflation, None,
                                       self._on_pallet_costmap_deflated)
            return

        self._sweep_idx += 1
        if self._sweep_idx >= len(self._sweep_stops):
            self._stop_sweep()
            self._state = MissionState.SWEEP_EMPTY
        else:
            self._state = MissionState.SWEEP_TURN_TO_STOP

    def _evaluate_sweep_stop(self) -> str | None:
        """Agrega las muestras de la parada actual: requiere mayoría de
        (pallet detectado AND QR detectado) y un contenido de QR no vacío
        consistente entre muestras. Devuelve el string del cliente si la
        parada es válida, o None si no hay pallet/QR confiable aquí."""
        positivos = [s for s in self._sweep_samples if s[0] and s[1] and s[2]]
        if len(positivos) <= len(self._sweep_samples) // 2:
            return None
        contenidos = [s[2] for s in positivos]
        # Requiere que el contenido decodificado sea consistente (evita QR mal leído)
        if len(set(contenidos)) != 1:
            self.get_logger().warn(f'QR inconsistente en parada {self._sweep_idx + 1}: {contenidos}')
            return None
        return contenidos[0]

    # ───────────────────────── Callbacks de visión ─────────────────────────

    def _cb_pallet_detected(self, msg: Bool):
        self._last_pallet_detected = msg.data

    def _cb_pallet_has_qr(self, msg: Bool):
        self._last_pallet_has_qr = msg.data

    def _cb_pallet_qr_content(self, msg: String):
        self._last_qr_content = msg.data

    # ───────────────────────── Alineación + FPGA ─────────────────────────

    def _send_deteccion_pallet(self, tipo: str):
        """Publica String('rack') o String('rodillo') en /deteccion_pallet.
        SOLO se llama desde SEND_DETECCION, es decir, después de confirmar
        self._goal_active == False tras llegar al waypoint del pallet —
        nunca antes (ver corrección de orden: evita que el /waypoint_reached
        de esa llegada sea malinterpretado por fpga_controller_node como
        'llegó al destino final')."""
        tipo_norm = 'rack' if tipo.startswith('rack') else 'rodillo'
        self._deteccion_pub.publish(String(data=tipo_norm))
        self.get_logger().info(f'Publicado /deteccion_pallet = "{tipo_norm}"')

    def _cb_alineacion(self, msg: Bool):
        """Vision confirma alineación fina completa ('ARRIVED'). Solo
        procesar en WAITING_ALIGNMENT. fpga_controller_node también
        escucha este topic y dispara su propia secuencia en paralelo —
        este nodo NO debe reenviar nada, solo usarlo como gatillo para
        avanzar su propia FSM hacia la espera calibrada de carga."""
        if self._state != MissionState.WAITING_ALIGNMENT or not msg.data:
            return
        self.get_logger().info('Alineación confirmada → esperando que la FPGA cargue el pallet')
        self._publish_external_cmd_active(False)
        self._fpga_wait_start = self.get_clock().now()
        self._state = MissionState.WAITING_LOAD

    def _area_actual(self) -> str:
        return _PICKUP_AREAS[self._area_idx]

    def _resolver_delivery(self, cliente: str) -> tuple[str, str]:
        """Mapea el nombre de cliente leído del QR a su clave en
        waypoints['delivery']. Asume que el QR contiene exactamente
        'cliente1'/'cliente2'/'cliente3' (igual que las claves del .yaml)."""
        if cliente not in self._waypoints['delivery']:
            raise RuntimeError(f"QR decodificado a '{cliente}' pero no existe en delivery del .yaml")
        return 'delivery', cliente

    def _resolver_acceso_cliente(self, cliente: str) -> tuple[str, str]:
        """Mapea el cliente a su punto de pre-alineación delivery.accesoc{N}
        (mismo N que clienteN) — ahí el robot ya queda orientado para entrar
        en línea recta a delivery[clienteN], evitando que rrt_node trace una
        curva de último momento dentro del pasillo angosto."""
        accesoc = cliente.replace('cliente', 'accesoc')
        if accesoc not in self._waypoints['delivery']:
            raise RuntimeError(f"falta 'delivery.{accesoc}' en waypoints.yaml para alinear hacia '{cliente}'")
        return 'delivery', accesoc

    # ───────────────────────── Bucle principal de la FSM ─────────────────────

    def _loop(self):
        """Despacha según self._state. Cada rama publica/transiciona y
        retorna; toda la lógica es asíncrona vía callbacks + este timer
        a 10 Hz — nunca bloquea el executor."""
        st = self._state

        if st == MissionState.WAITING_LOCALIZATION:
            return  # arranca por _cb_mcl_wandering

        if self._goal_active and self._goal_start_time is not None:
            if self._elapsed_s(self._goal_start_time) > self._nav_timeout:
                self.get_logger().error(
                    f'Timeout de navegación en estado {st.name} → abortando misión')
                self._cancel_pub.publish(Empty())
                self._state = MissionState.ABORTED
                return

        # ---- Navegación a waypoint 'general' del área activa ----
        if st == MissionState.NAV_TO_AREA_GENERAL:
            if self._ensure_nav_goal(self._area_actual(), 'general'):
                yaw = self._yaw_de_waypoint(self._area_actual(), 'general')
                self._start_align_to_yaw(yaw, self._on_aligned_to_general)
            return

        if st == MissionState.ALIGN_TO_YAW:
            self._do_align_to_yaw()
            return

        if st == MissionState.SWEEP_TURN_TO_STOP:
            self._do_sweep_turn()
            return

        if st == MissionState.SWEEP_SAMPLING:
            self._do_sweep_sampling()
            return

        if st == MissionState.SWEEP_EMPTY:
            self._handle_sweep_empty()
            return

        if st == MissionState.NAV_TO_PALLET:
            if self._ensure_nav_goal(self._area_actual(), f'p{self._pallet_idx_objetivo}'):
                yaw = self._yaw_de_waypoint(self._area_actual(), f'p{self._pallet_idx_objetivo}')
                self._start_align_to_yaw(yaw, self._on_aligned_to_pallet)
            return

        if st == MissionState.SEND_DETECCION:
            tipo = self._area_actual()
            self._send_deteccion_pallet(tipo)
            # align_and_approach/tracking publican directo a /cmd_vel (no pasan
            # por dwa_node) durante la alineación fina — avisamos para que
            # dwa_node ceda el control y no compitan por /cmd_vel al mismo
            # tiempo (mismo patrón que el barrido / ALIGN_TO_YAW). Se apaga en
            # _cb_alineacion al confirmarse la alineación.
            self._publish_external_cmd_active(True)
            self._state = MissionState.WAITING_ALIGNMENT
            return

        if st == MissionState.WAITING_ALIGNMENT:
            return  # avanza por _cb_alineacion

        if st == MissionState.WAITING_LOAD:
            if self._elapsed_s(self._fpga_wait_start) >= self._fpga_settle:
                self._fpga_wait_start = None
                self._start_costmap_restore(self._on_pallet_costmap_restored)
            return

        # ---- Secuencia de entrega (acceso -> desinflar -> cliente -> regresar -> inflar) ----
        if st == MissionState.NAV_TO_DELIVERY_ACCESO:
            if self._ensure_nav_goal('delivery', 'acceso'):
                self._start_costmap_reduce(self._delivery_inflation,
                                           self._delivery_robot_radius,
                                           self._on_costmap_deflated)
            return

        if st == MissionState.DEFLATE_COSTMAP:
            self._poll_costmap_inflation()
            return

        if st == MissionState.NAV_TO_DELIVERY_ALIGN:
            try:
                area_a, punto_a = self._resolver_acceso_cliente(self._cliente_actual)
            except RuntimeError as e:
                self.get_logger().error(f'{e} → abortando misión (QR mal leído o inesperado)')
                self._state = MissionState.ABORTED
                return
            if self._ensure_nav_goal(area_a, punto_a):
                self._state = MissionState.NAV_TO_DELIVERY
            return

        if st == MissionState.NAV_TO_DELIVERY:
            try:
                area_d, punto_d = self._resolver_delivery(self._cliente_actual)
            except RuntimeError as e:
                self.get_logger().error(f'{e} → abortando misión (QR mal leído o inesperado)')
                self._state = MissionState.ABORTED
                return
            if self._ensure_nav_goal(area_d, punto_d):
                self._on_arrived_delivery()
                self._state = MissionState.WAITING_UNLOAD
            return

        if st == MissionState.WAITING_UNLOAD:
            if self._elapsed_s(self._fpga_wait_start) >= self._fpga_settle:
                self._fpga_wait_start = None
                self._start_reverse_to_acceso(self._on_arrived_acceso_reversing)
            return

        if st == MissionState.REVERSE_TO_ACCESO:
            self._do_reverse_to_acceso()
            return

        if st == MissionState.INFLATE_COSTMAP:
            self._poll_costmap_inflation()
            return

        if st in (MissionState.MISSION_COMPLETE, MissionState.ABORTED):
            return  # estado terminal

    # ── helpers de transición ────────────────────────────────────────────

    def _ensure_nav_goal(self, area: str, punto: str) -> bool:
        """Envía el goal a (area, punto) UNA SOLA VEZ (lo recuerda en
        self._nav_pending_target) y devuelve True exactamente en el ciclo
        en que /waypoint_reached confirma la llegada a ESE goal — momento
        en que el llamador debe transicionar de estado. Evita reenvíos
        mientras el goal sigue en curso y falsos positivos de llegadas
        viejas."""
        target = (area, punto)

        if self._nav_pending_target is None:
            self._send_nav_goal(self._waypoint_to_pose(area, punto))
            self._nav_pending_target = target
            return False

        if self._nav_pending_target == target and not self._goal_active:
            self._nav_pending_target = None
            return True

        return False

    def _on_arrived_general(self):
        self._start_sweep(self._area_actual())

    def _on_aligned_to_general(self):
        """Termina el giro hacia el yaw grabado de 'general' → arranca el barrido."""
        self._on_arrived_general()
        self._state = MissionState.SWEEP_TURN_TO_STOP

    def _on_aligned_to_pallet(self):
        """Termina el giro hacia el yaw grabado de 'p{N}' → publica /deteccion_pallet."""
        self._state = MissionState.SEND_DETECCION

    def _on_costmap_deflated(self):
        """Costmap ya desinflado y asentado → ahora sí hay ruta hacia la
        zona de clientes; primero pasa por delivery.accesoc{N} para quedar
        pre-alineado y entrar derecho a delivery[cliente] (ver
        NAV_TO_DELIVERY_ALIGN / _resolver_acceso_cliente)."""
        self._state = MissionState.NAV_TO_DELIVERY_ALIGN

    def _on_costmap_inflated(self):
        """Costmap restaurado a su inflado normal, robot de vuelta en
        delivery.acceso → continuar con la siguiente área."""
        self._advance_to_next_area()

    def _on_pallet_costmap_deflated(self):
        """Costmap ya desinflado para esta área → ahora sí hay ruta hacia
        el waypoint p{N} del pallet elegido."""
        self._state = MissionState.NAV_TO_PALLET

    def _on_pallet_costmap_restored(self):
        """Costmap restaurado a su inflado normal tras cargar el pallet
        (el robot ya está detenido junto al rack, no hace falta seguir con
        el margen reducido) → continuar hacia delivery.acceso."""
        self._state = MissionState.NAV_TO_DELIVERY_ACCESO

    def _on_arrived_acceso_reversing(self):
        """Llegó retrocediendo a delivery.acceso → ya se puede reinflar."""
        self._start_costmap_restore(self._on_costmap_inflated)

    def _on_arrived_delivery(self):
        self._fpga_wait_start = self.get_clock().now()
        self.get_logger().info(
            f'{self._area_actual().upper()} completado → cliente={self._cliente_actual} '
            f'| esperando que la FPGA deposite el pallet')

    def _handle_sweep_empty(self):
        """Ningún pallet con QR en el área actual. Para 'rodillos' esto no
        debería ocurrir (la tarea siempre arranca ahí con pallet asegurado);
        loggear como advertencia y abortar. Para racks, avanzar al
        siguiente; si era rack3, declarar misión completa sin carga."""
        area = self._area_actual()
        if area == 'rodillos':
            self.get_logger().error('RODILLOS sin pallet con QR — situación inesperada, abortando')
            self._state = MissionState.ABORTED
            return

        self.get_logger().info(f'{area} vacío → siguiente área')
        if self._area_idx + 1 >= len(_PICKUP_AREAS):
            self.get_logger().info('Los tres racks están vacíos → misión completa sin carga pendiente')
            self._state = MissionState.MISSION_COMPLETE
        else:
            self._area_idx += 1
            self._state = MissionState.NAV_TO_AREA_GENERAL

    def _advance_to_next_area(self):
        self._cliente_actual = None
        self._pallet_idx_objetivo = None
        if self._area_idx + 1 >= len(_PICKUP_AREAS):
            self.get_logger().info('Misión completa — todas las áreas procesadas ✓')
            self._state = MissionState.MISSION_COMPLETE
        else:
            self._area_idx += 1
            self._state = MissionState.NAV_TO_AREA_GENERAL


def main(args=None):
    rclpy.init(args=args)
    node = MissionManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
