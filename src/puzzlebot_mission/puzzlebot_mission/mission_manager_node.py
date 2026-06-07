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

    NAV_TO_PALLET     = auto()       # navegando al waypoint p{N} del pallet elegido
    SEND_DETECCION    = auto()       # publicar /deteccion_pallet (solo tras llegar al pallet)
    WAITING_ALIGNMENT = auto()       # esperando /alineation/booleano == True
    WAITING_LOAD      = auto()       # espera calibrada: FPGA terminando de cargar
    NAV_TO_DELIVERY   = auto()       # navegando al waypoint delivery[cliente]
    WAITING_UNLOAD    = auto()       # espera calibrada: FPGA terminando de depositar

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

        self._sweep_range   = math.radians(self.get_parameter('sweep_range_deg').value)
        self._sweep_speed   = self.get_parameter('sweep_angular_speed').value
        self._sweep_align_th = math.radians(self.get_parameter('sweep_align_threshold_deg').value)
        self._sweep_settle  = self.get_parameter('sweep_settle_time_s').value
        self._sweep_n_samples = int(self.get_parameter('sweep_samples_per_stop').value)
        self._nav_timeout   = self.get_parameter('nav_timeout_s').value
        self._fpga_settle   = self.get_parameter('fpga_settle_time_s').value

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

        self._rth = 0.0                       # yaw actual (de /mcl_pose)
        self._pose_ready = False

        self._sweep_stops: list[float] = []
        self._sweep_idx = 0
        self._sweep_phase_start = None         # Time: inicio del asentamiento de la parada actual
        self._sweep_samples: list[tuple[bool, bool, str | None]] = []

        self._align_target_yaw: float | None = None
        self._align_on_done = None              # callable() invocado al terminar el giro

        self._fpga_wait_start = None            # Time: inicio de la espera calibrada de FPGA

        # ---- Suscripciones a estado del sistema ----
        self.create_subscription(Bool, '/mcl_wandering', self._cb_mcl_wandering, 10)
        self.create_subscription(Empty, '/waypoint_reached', self._cb_waypoint_reached, _RELIABLE)
        from geometry_msgs.msg import PoseWithCovarianceStamped
        self.create_subscription(
            PoseWithCovarianceStamped, '/mcl_pose', self._cb_mcl_pose, _RELIABLE)

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
        self._cmd_vel_pub   = self.create_publisher(Twist, '/cmd_vel', _RELIABLE)
        # Bandera para que dwa_node ceda /cmd_vel mientras este nodo gira durante
        # el barrido (replica el patrón de /mcl_wandering). REQUIERE modificar
        # dwa_node para que la respete — ver advertencia pendiente.
        self._ext_cmd_pub = self.create_publisher(Bool, '/external_cmd_vel_active', 10)

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
        q = msg.pose.pose.orientation
        self._rth = _quat_to_yaw(q.w, q.z)
        self._pose_ready = True

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
            self._state = MissionState.NAV_TO_PALLET
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
            self._state = MissionState.WAITING_ALIGNMENT
            return

        if st == MissionState.WAITING_ALIGNMENT:
            return  # avanza por _cb_alineacion

        if st == MissionState.WAITING_LOAD:
            if self._elapsed_s(self._fpga_wait_start) >= self._fpga_settle:
                self._fpga_wait_start = None
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
                self._advance_to_next_area()
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
