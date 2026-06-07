#!/usr/bin/env python3
"""
vision_faker.py — Publica señales falsas de visión para probar mission_manager
en simulación, donde Gazebo no tiene pallets ni QR reales (son sólidos
rectangulares / paredes).

Topics publicados (mismos nombres que align_and_approach/tracking, así que
mission_manager no distingue entre la fuente real y esta simulada):
  /pallet_detected      (std_msgs/Bool)
  /pallet_has_qr        (std_msgs/Bool)
  /pallet_qr_content    (std_msgs/String)
  /alineation/booleano  (std_msgs/Bool)

Comandos (terminal interactiva):
  pallet clienteN   -> simula pallet+QR detectados con contenido 'clienteN'
                       (p.ej. 'pallet cliente2') — escríbelo cuando el robot
                       esté detenido en la parada de barrido que quieres que
                       "tenga" el pallet
  nopallet          -> simula parada vacía (sin pallet/QR) — para las demás
                       paradas del barrido
  align on          -> simula que /alineation/booleano confirmó la alineación
                       fina (avanza mission_manager de WAITING_ALIGNMENT a
                       WAITING_LOAD) — escríbelo cuando el robot llegue al
                       waypoint del pallet
  align off         -> vuelve a 'sin confirmar' (por si quieres reiniciar la prueba)
  salir             -> termina

Reinicio automático entre áreas: mission_manager anuncia el inicio de cada
barrido en /mission_sweep_area (ver _start_sweep) — este nodo lo escucha y
reinicia solo su estado simulado a 'sin pallet / sin alinear' cada vez que
empieza un área nueva, para no arrastrar por accidente el "pallet cliente2"
de un área a la siguiente (como pasaba al probar manualmente la búsqueda
rack1->rack2->rack3). Sigue pudiéndose sobreescribir con los comandos de
arriba en cualquier momento, igual que antes.

Uso:
  ros2 run puzzlebot_mission vision_faker
"""
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Bool, String


_RELIABLE = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


class VisionFaker(Node):

    def __init__(self):
        super().__init__('vision_faker')

        self._det_pub = self.create_publisher(Bool, '/pallet_detected', _RELIABLE)
        self._qr_flag_pub = self.create_publisher(Bool, '/pallet_has_qr', _RELIABLE)
        self._qr_content_pub = self.create_publisher(String, '/pallet_qr_content', _RELIABLE)
        self._align_pub = self.create_publisher(Bool, '/alineation/booleano', _RELIABLE)

        self._detected = False
        self._has_qr = False
        self._qr_content = ''
        self._aligned = False

        # Reinicio automático al cambiar de área (ver _cb_sweep_area) — evita
        # arrastrar por accidente el último "pallet clienteN"/"align on" de
        # un área a la siguiente cuando se prueba manualmente la búsqueda
        # rack1->rack2->rack3.
        self.create_subscription(String, '/mission_sweep_area', self._cb_sweep_area, _RELIABLE)

        # Publica el estado actual a 10 Hz — mission_manager solo recuerda el
        # último valor recibido de cada topic, pero conviene refrescarlo
        # seguido para que las muestras del barrido (5 lecturas a 10 Hz tras
        # el asentamiento) siempre vean el valor vigente.
        self.create_timer(0.1, self._publish_current_state)

        self.get_logger().info(
            "vision_faker listo | comandos: 'pallet clienteN', 'nopallet', "
            "'align on'/'align off', 'salir'")

        self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._input_thread.start()

    def _cb_sweep_area(self, msg: String):
        """mission_manager anunció el inicio de un barrido nuevo (ver
        _start_sweep) → reinicia el estado simulado a 'sin pallet / sin
        alinear', como si se hubiera mandado 'nopallet' + 'align off'.
        Así cada área arranca en blanco y hay que comandar explícitamente
        el pallet (y la alineación) que se quiera simular para ESA área,
        sin arrastrar el de la anterior por descuido."""
        if not (self._detected or self._has_qr or self._qr_content or self._aligned):
            return  # ya estaba en blanco — no ensuciar la consola con ruido
        self._detected = False
        self._has_qr = False
        self._qr_content = ''
        self._aligned = False
        self.get_logger().info(
            f'-> Área "{msg.data}": estado de visión reiniciado automáticamente '
            f'(sin pallet, sin alinear) — usa "pallet clienteN"/"align on" si quieres simularlo aquí')

    def _publish_current_state(self):
        self._det_pub.publish(Bool(data=self._detected))
        self._qr_flag_pub.publish(Bool(data=self._has_qr))
        self._qr_content_pub.publish(String(data=self._qr_content))
        self._align_pub.publish(Bool(data=self._aligned))

    def _input_loop(self):
        while rclpy.ok():
            try:
                line = input('vision_faker> ').strip()
            except EOFError:
                break
            if not line:
                continue
            self._handle_command(line)

    def _handle_command(self, line: str):
        partes = line.lower().split()
        cmd = partes[0]

        if cmd in ('salir', 'exit', 'quit'):
            self.get_logger().info('Cerrando vision_faker...')
            rclpy.shutdown()
            return

        if cmd == 'pallet' and len(partes) == 2:
            self._detected = True
            self._has_qr = True
            self._qr_content = partes[1]
            self.get_logger().info(f'-> Simulando pallet+QR detectado: "{self._qr_content}"')
            return

        if cmd == 'nopallet':
            self._detected = False
            self._has_qr = False
            self._qr_content = ''
            self.get_logger().info('-> Simulando parada vacía (sin pallet/QR)')
            return

        if cmd == 'align' and len(partes) == 2 and partes[1] in ('on', 'off'):
            self._aligned = (partes[1] == 'on')
            self.get_logger().info(f'-> Simulando alineación = {self._aligned}')
            return

        self.get_logger().warn(
            f"Comando no reconocido: '{line}'. Usa 'pallet clienteN', 'nopallet', "
            "'align on'/'align off' o 'salir'.")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VisionFaker()
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
