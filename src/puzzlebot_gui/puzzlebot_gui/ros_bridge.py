import math
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, PoseArray
from sensor_msgs.msg import JointState, CompressedImage
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import Twist

_DEAD_TIMEOUT = 5.0

_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)
_RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)
_LATCHED_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def _quat_to_yaw(x, y, z, w):
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


def _yaw_to_quat(yaw):
    return math.cos(yaw / 2.0), math.sin(yaw / 2.0)


class RosBridge(Node):
    """
    Nodo ROS 2 puente. Solo monitoreo + envío de waypoints.
    Sin teleop ni control de navegación.
    """

    def __init__(self):
        super().__init__('puzzlebot_gui_bridge')
        self._lock = threading.Lock()

        # Estado interno
        self._odom_pose  = None   # {x, y, yaw, vl, va, t} — fuente de velocidades
        self._mcl_pose   = None   # {x, y, yaw, t}          — fuente de posición (más precisa)
        self._costmap    = None
        self._static_map = None
        self._path       = None
        self._camera     = None

        # Waypoint sequencing
        self._waypoints      = []
        self._wp_index       = 0
        self._wp_reach_dist  = 0.30  # metros para considerar un WP alcanzado

        # Liveness de nodos — solo los que publican en el stack actual
        # Removidos: control (/goal_reached=0pub), setpoint_generator (/set_point=0pub),
        #            icp_node (/icp_odom=0pub), icp_map_node (/icp_map=0pub), scan_relay
        self._node_stamps = {
            'localisation':             0.0,   # /odom
            'mcl_node':                 0.0,   # /mcl_pose, /particle_cloud, /mcl_converged
            'rrt_node':                 0.0,   # /global_path
            'dwa_node':                 0.0,   # /cmd_vel
            'path_follower_node':       0.0,   # /cmd_vel_reference
            'costmap_node':             0.0,   # /costmap
            'joint_vel_bridge':         0.0,   # /wr
            'joint_state_publisher':    0.0,   # /joint_states
            'active_localization_node': 0.0,   # /mcl_wandering
        }

        # Suscripciones — solo tópicos con publishers activos en el stack actual
        self.create_subscription(Odometry,                  '/odom',              self._odom_cb,          _SENSOR_QOS)
        self.create_subscription(PoseWithCovarianceStamped, '/mcl_pose',          self._mcl_pose_cb,      _RELIABLE_QOS)
        self.create_subscription(OccupancyGrid,             '/map',               self._map_static_cb,    _LATCHED_QOS)
        self.create_subscription(OccupancyGrid,             '/costmap',           self._costmap_cb,       _LATCHED_QOS)
        self.create_subscription(Path,                      '/global_path',       self._path_cb,          _RELIABLE_QOS)
        self.create_subscription(PoseArray,                 '/particle_cloud',    self._particles_cb,     _RELIABLE_QOS)
        self.create_subscription(Twist,                     '/cmd_vel',           self._cmd_vel_cb,       _RELIABLE_QOS)
        self.create_subscription(Twist,                     '/cmd_vel_reference', self._cmd_vel_ref_cb,   _RELIABLE_QOS)
        self.create_subscription(Bool,                      '/mcl_converged',     self._mcl_converged_cb, _RELIABLE_QOS)
        self.create_subscription(Bool,                      '/mcl_wandering',     self._wandering_cb,     _RELIABLE_QOS)
        self.create_subscription(JointState,                '/joint_states',      self._joint_states_cb,  _RELIABLE_QOS)
        self.create_subscription(Float32,                   '/wr',                self._wr_cb,            _SENSOR_QOS)
        self.create_subscription(CompressedImage,           '/align/compressed',  self._image_cb,         _SENSOR_QOS)

        # Publicadores — solo waypoints
        self._pub_goal = self.create_publisher(PoseStamped, '/goal_pose', _RELIABLE_QOS)

        self.get_logger().info('RosBridge listo')

    # ── Callbacks de suscripciones ────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        now = time.time()
        q   = msg.pose.pose.orientation
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)
        with self._lock:
            self._odom_pose = {
                'x': x, 'y': y, 'yaw': yaw,
                'vel_linear':  msg.twist.twist.linear.x,
                'vel_angular': msg.twist.twist.angular.z,
                'timestamp':   now,
            }
            self._node_stamps['localisation'] = now
        self._check_waypoint_advance(x, y)

    def _mcl_pose_cb(self, msg: PoseWithCovarianceStamped):
        now = time.time()
        q   = msg.pose.pose.orientation
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        with self._lock:
            self._mcl_pose = {
                'x': x, 'y': y,
                'yaw':       _quat_to_yaw(q.x, q.y, q.z, q.w),
                'timestamp': now,
            }
            self._node_stamps['mcl_node'] = now
        self._check_waypoint_advance(x, y)

    def _map_static_cb(self, msg: OccupancyGrid):
        q = msg.info.origin.orientation
        # Convertir -1 (desconocido) a 127: en protobuf int32, -1 ocupa 10 bytes
        # mientras que 127 ocupa 1 byte. Reduce el mensaje de ~2.6 MB a ~260 KB.
        clean = [v if v >= 0 else 127 for v in msg.data]
        with self._lock:
            self._static_map = {
                'data':       clean,
                'width':      msg.info.width,
                'height':     msg.info.height,
                'resolution': msg.info.resolution,
                'origin_x':   msg.info.origin.position.x,
                'origin_y':   msg.info.origin.position.y,
                'origin_yaw': _quat_to_yaw(q.x, q.y, q.z, q.w),
            }
        self.get_logger().info(
            f'Mapa estático recibido: {msg.info.width}x{msg.info.height} celdas'
        )

    def _costmap_cb(self, msg: OccupancyGrid):
        now = time.time()
        clean = [v if v >= 0 else 0 for v in msg.data]
        with self._lock:
            o = self._odom_pose
            self._costmap = {
                'data':       clean,
                'width':      msg.info.width,
                'height':     msg.info.height,
                'resolution': msg.info.resolution,
                'origin_x':   msg.info.origin.position.x,
                'origin_y':   msg.info.origin.position.y,
                'robot_x':    o['x']   if o else 0.0,
                'robot_y':    o['y']   if o else 0.0,
                'robot_yaw':  o['yaw'] if o else 0.0,
                'timestamp':  now,
            }
            self._node_stamps['costmap_node'] = now

    def _path_cb(self, msg: Path):
        now = time.time()
        poses = []
        for p in msg.poses:
            q = p.pose.orientation
            poses.append((
                p.pose.position.x,
                p.pose.position.y,
                _quat_to_yaw(q.x, q.y, q.z, q.w),
            ))
        with self._lock:
            prev = self._path.get('particle_poses', []) if self._path else []
            self._path = {'poses': poses, 'particle_poses': prev, 'timestamp': now}
            self._node_stamps['rrt_node'] = now

    def _particles_cb(self, msg: PoseArray):
        now = time.time()
        step = max(1, len(msg.poses) // 200)
        particles = []
        for p in msg.poses[::step]:
            q = p.orientation
            particles.append((p.position.x, p.position.y, _quat_to_yaw(q.x, q.y, q.z, q.w)))
        with self._lock:
            if self._path is None:
                self._path = {'poses': [], 'particle_poses': particles, 'timestamp': now}
            else:
                # Actualizar timestamp para que StreamPath detecte el cambio
                self._path['particle_poses'] = particles
                self._path['timestamp'] = now
            self._node_stamps['mcl_node'] = now

    def _cmd_vel_cb(self, msg: Twist):
        with self._lock:
            self._node_stamps['dwa_node'] = time.time()

    def _cmd_vel_ref_cb(self, _):
        with self._lock:
            self._node_stamps['path_follower_node'] = time.time()

    def _mcl_converged_cb(self, _):
        with self._lock:
            self._node_stamps['mcl_node'] = time.time()

    def _wandering_cb(self, _):
        with self._lock:
            self._node_stamps['active_localization_node'] = time.time()

    def _joint_states_cb(self, _):
        with self._lock:
            self._node_stamps['joint_state_publisher'] = time.time()

    def _wr_cb(self, _):
        with self._lock:
            self._node_stamps['joint_vel_bridge'] = time.time()

    def _image_cb(self, msg: CompressedImage):
        # tracking.py ya publica el JPEG con las detecciones YOLO y la línea de
        # alineación dibujadas, así que solo se reenvía tal cual (sin recodificar).
        if not msg.data:
            return
        with self._lock:
            self._camera = {
                'jpeg_data':  bytes(msg.data),
                'detections': [],
                'width':      0,
                'height':     0,
                'timestamp':  time.time(),
            }

    # ── Waypoints ─────────────────────────────────────────────────────────────

    def publish_waypoints(self, waypoints: list) -> bool:
        if not waypoints:
            return False
        with self._lock:
            self._waypoints = list(waypoints)
            self._wp_index  = 0
        ok = self._publish_goal(waypoints[0])
        if ok:
            self.get_logger().info(
                f'Waypoints cargados ({len(waypoints)}). Navegando a WP[0]: '
                f'({waypoints[0][0]:.2f}, {waypoints[0][1]:.2f})'
            )
        return ok

    def _publish_goal(self, wp) -> bool:
        try:
            msg = PoseStamped()
            msg.header.frame_id = 'map'
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.pose.position.x = float(wp[0])
            msg.pose.position.y = float(wp[1])
            qw, qz = _yaw_to_quat(float(wp[2]))
            msg.pose.orientation.w = qw
            msg.pose.orientation.z = qz
            self._pub_goal.publish(msg)
            return True
        except Exception as exc:
            self.get_logger().error(f'Error publicando goal: {exc}')
            return False

    def _check_waypoint_advance(self, robot_x: float, robot_y: float):
        """Avanza al siguiente waypoint cuando el robot está suficientemente cerca."""
        with self._lock:
            wps = self._waypoints
            idx = self._wp_index
            if not wps or idx >= len(wps):
                return
            wp   = wps[idx]
            dist = math.sqrt((robot_x - wp[0])**2 + (robot_y - wp[1])**2)
            if dist > self._wp_reach_dist:
                return
            self._wp_index = idx + 1
            next_idx = self._wp_index
            next_wp  = wps[next_idx] if next_idx < len(wps) else None

        if next_wp:
            self._publish_goal(next_wp)
            self.get_logger().info(
                f'WP[{idx}] alcanzado. Navegando a WP[{next_idx}]: '
                f'({next_wp[0]:.2f}, {next_wp[1]:.2f})'
            )
        else:
            self.get_logger().info('Todos los waypoints alcanzados.')

    # ── Getters thread-safe ───────────────────────────────────────────────────

    def get_telemetry(self):
        """
        Combina odom (velocidades) y MCL (posición).
        Si MCL no está disponible o es viejo (> 0.5 s), usa posición de odom.
        """
        with self._lock:
            if not self._odom_pose:
                return None
            o = self._odom_pose
            # Posición: preferir MCL si llegó hace menos de 0.5 s
            m = self._mcl_pose
            if m and (time.time() - m['timestamp']) < 0.5:
                pos_x, pos_y, yaw = m['x'], m['y'], m['yaw']
            else:
                pos_x, pos_y, yaw = o['x'], o['y'], o['yaw']
            return {
                'pos_x':       pos_x,
                'pos_y':       pos_y,
                'yaw':         yaw,
                'vel_linear':  o['vel_linear'],
                'vel_angular': o['vel_angular'],
                'timestamp':   o['timestamp'],
            }

    def get_costmap(self):
        with self._lock:
            # Sin mapa estático ni costmap no hay nada que enviar
            if not self._costmap and not self._static_map:
                return None

            if self._costmap:
                result = dict(self._costmap)
            else:
                # Costmap_node no está corriendo; enviamos solo el mapa estático
                o = self._odom_pose
                result = {
                    'data': [], 'width': 0, 'height': 0,
                    'resolution': 0.05,
                    'origin_x': 0.0, 'origin_y': 0.0,
                    'robot_x':   o['x']   if o else 0.0,
                    'robot_y':   o['y']   if o else 0.0,
                    'robot_yaw': o['yaw'] if o else 0.0,
                    'timestamp': time.time(),
                }

            if self._static_map:
                result['map_data']       = self._static_map['data']
                result['map_width']      = self._static_map['width']
                result['map_height']     = self._static_map['height']
                result['map_resolution'] = self._static_map['resolution']
                result['map_origin_x']   = self._static_map['origin_x']
                result['map_origin_y']   = self._static_map['origin_y']
                result['map_origin_yaw'] = self._static_map.get('origin_yaw', 0.0)
            else:
                result['map_data'] = []
            return result

    def get_path(self):
        with self._lock:
            return dict(self._path) if self._path else None

    def get_camera_frame(self):
        with self._lock:
            return dict(self._camera) if self._camera else None

    def get_node_status(self) -> list:
        now = time.time()
        with self._lock:
            return [
                {
                    'name':      name,
                    'alive':     (now - stamp) < _DEAD_TIMEOUT and stamp > 0.0,
                    'last_seen': stamp,
                }
                for name, stamp in self._node_stamps.items()
            ]
