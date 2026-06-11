#!/usr/bin/env python3
"""
waypoint_recorder.py — Captura interactiva de waypoints para waypoints.yaml.

Flujo de uso:
  1. Levanta MCL en simulación (publica /mcl_pose) y maneja el robot
     (teleop, RViz "2D Nav Goal", etc.) hasta la posición que quieres marcar.
  2. Corre este nodo pasándole la ruta al waypoints.yaml:
       ros2 run puzzlebot_mission waypoint_recorder \\
           --ros-args -p yaml_path:=/ruta/a/waypoints.yaml
  3. En la terminal escribe la clave del punto, p.ej.:
       rodillos.general
       rodillos.p3
       delivery.cliente1
     y Enter — captura la pose actual de /mcl_pose, la convierte a quaternion
     yaw-only y la escribe directo en el .yaml (sobreescribe esa entrada si
     ya existía).
  4. 'salir' (o Ctrl+C) para terminar.
"""
import math
import threading

import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped


_RELIABLE = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


def _quat_to_yaw(w: float, z: float) -> float:
    return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)


def _yaw_to_quat_wz(yaw: float) -> tuple[float, float]:
    return math.cos(yaw / 2.0), math.sin(yaw / 2.0)


class _FlowMap(dict):
    """Marca un dict para serializarlo en estilo inline {x: 0.0, y: 0.0}."""
    pass


def _represent_flow_map(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=True)


yaml.add_representer(_FlowMap, _represent_flow_map, Dumper=yaml.SafeDumper)


def _flowify(data: dict) -> dict:
    """Envuelve position/orientation de cada waypoint en _FlowMap para
    reescribir todo el archivo en estilo inline."""
    out = {}
    for area, puntos in data.items():
        out[area] = {}
        for punto, wp in puntos.items():
            out[area][punto] = {
                'position': _FlowMap(wp['position']),
                'orientation': _FlowMap(wp['orientation']),
            }
    return out


def _read_header(path: str) -> str:
    """Conserva el bloque de comentarios inicial del .yaml (yaml.safe_dump no los preserva)."""
    lines = []
    try:
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    lines.append(line)
                else:
                    break
    except FileNotFoundError:
        return ''
    return ''.join(lines)


class WaypointRecorder(Node):

    def __init__(self):
        super().__init__('waypoint_recorder')
        self.declare_parameter('yaml_path', '')
        self.declare_parameter('pose_topic', '/mcl_pose')

        self._yaml_path = self.get_parameter('yaml_path').value
        if not self._yaml_path:
            raise RuntimeError("Parámetro 'yaml_path' vacío — pasa la ruta a waypoints.yaml")

        pose_topic = self.get_parameter('pose_topic').value
        self._header = _read_header(self._yaml_path)
        self._latest_pose = None

        self.create_subscription(PoseWithCovarianceStamped, pose_topic, self._cb_pose, _RELIABLE)

        self.get_logger().info(
            f'waypoint_recorder listo | escuchando {pose_topic} | guardando en {self._yaml_path}')
        self.get_logger().info(
            "Maneja el robot a la posición deseada y escribe la clave del waypoint "
            "(p.ej. 'rodillos.p3', 'delivery.cliente1') + Enter para capturarla. "
            "'salir' para terminar.")

        self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._input_thread.start()

    def _cb_pose(self, msg: PoseWithCovarianceStamped):
        self._latest_pose = msg.pose.pose

    def _input_loop(self):
        while rclpy.ok():
            try:
                line = input('waypoint> ').strip()
            except EOFError:
                break
            if not line:
                continue
            if line.lower() in ('salir', 'exit', 'quit'):
                self.get_logger().info('Cerrando waypoint_recorder...')
                rclpy.shutdown()
                break
            self._record(line)

    def _record(self, clave: str):
        partes = clave.split('.')
        if len(partes) != 2 or not all(partes):
            self.get_logger().warn(
                f"Formato inválido '{clave}' — usa 'area.punto', p.ej. 'rodillos.p3'")
            return
        area, punto = partes

        pose = self._latest_pose
        if pose is None:
            self.get_logger().warn('Todavía no llega ninguna pose — ¿está corriendo MCL?')
            return

        yaw = _quat_to_yaw(pose.orientation.w, pose.orientation.z)
        w, z = _yaw_to_quat_wz(yaw)

        try:
            with open(self._yaml_path, 'r') as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            data = {}

        data.setdefault(area, {})[punto] = {
            'position': {
                'x': round(float(pose.position.x), 3),
                'y': round(float(pose.position.y), 3),
            },
            'orientation': {
                'w': round(w, 4),
                'z': round(z, 4),
            },
        }

        with open(self._yaml_path, 'w') as f:
            f.write(self._header)
            yaml.dump(_flowify(data), f, Dumper=yaml.SafeDumper,
                      default_flow_style=False, sort_keys=False, allow_unicode=True)

        self.get_logger().info(
            f'Guardado {area}.{punto} → x={pose.position.x:.3f} y={pose.position.y:.3f} '
            f'yaw={math.degrees(yaw):.1f}°')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = WaypointRecorder()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
