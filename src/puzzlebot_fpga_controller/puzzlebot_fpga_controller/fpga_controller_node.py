import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Empty
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import ctypes
import os

# ─────────────────────────────────────
# Cargar librería C compilada como .so
# ─────────────────────────────────────
LIB_PATH = os.path.join(os.path.dirname(__file__), 'libspi_fsm.so')
lib = ctypes.CDLL(LIB_PATH)

lib.spi_init.restype  = ctypes.c_int
lib.spi_init.argtypes = []

lib.spi_get_estado.restype  = ctypes.c_uint8
lib.spi_get_estado.argtypes = []

lib.esperar_estado.restype  = ctypes.c_int
lib.esperar_estado.argtypes = [ctypes.c_uint8, ctypes.c_int]

lib.spi_send_secuencia.restype  = None
lib.spi_send_secuencia.argtypes = [
    ctypes.c_uint8,   # cmd
    ctypes.c_uint16,  # t_subir
    ctypes.c_uint16,  # t_subir_poco
    ctypes.c_uint16,  # t_delay
    ctypes.c_uint16,  # t_bajar_vision
    ctypes.c_uint16,  # t_bajar_final
]

lib.spi_send_command.restype  = None
lib.spi_send_command.argtypes = [ctypes.c_uint8]

# Comandos
CMD_RACK     = 0x10
CMD_RODILLO  = 0x11
CMD_ALINEADO = 0x21
CMD_WAYPOINT = 0x22

# Estados FPGA
STATE_IDLE               = 0x00
STATE_ESPERAR_ALINEACION = 0x02
STATE_ESPERAR_WAYPOINT   = 0x06

# ── Tiempos calibrados RACK ──
# RACK va directo a ESPERAR_ALINEACION, t_subir=0
T_RACK_SUBIR        = 0
T_RACK_SUBIR_POCO   = 1200
T_RACK_DELAY        = 3000
T_RACK_BAJAR_VISION = 1700
T_RACK_BAJAR_FINAL  = 2600

# ── Tiempos calibrados RODILLO ──
# RODILLO sube primero con t_subir
T_ROD_SUBIR        = 1500
T_ROD_SUBIR_POCO   = 500
T_ROD_DELAY        = 3000
T_ROD_BAJAR_VISION = 7200
T_ROD_BAJAR_FINAL  = 400

TIMEOUT = 500  # 500 x 100ms = 50 segundos


class FpgaControllerNode(Node):

    def __init__(self):
        super().__init__('fpga_controller_node')

        if lib.spi_init() < 0:
            self.get_logger().error('Error inicializando SPI')
            raise RuntimeError('SPI init failed')
        self.get_logger().info('SPI inicializado correctamente')

        self._tipo_actual      = None
        self._alineado         = False
        self._secuencia_activa = False

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # Detección rack o rodillo desde nodo de visión
        self.create_subscription(
            String, '/deteccion_pallet',
            self._cb_deteccion, qos)

        # Alineación completada
        self.create_subscription(
            Bool, '/alineation/booleano',
            self._cb_alineacion, qos)

        # Waypoint alcanzado desde nav stack
        self.create_subscription(
            Empty, '/waypoint_reached',
            self._cb_waypoint_reached, qos)

        self.get_logger().info('Nodo FPGA listo — esperando detección de pallet')

    # ─────────────────────────────────────
    # Detección rack o rodillo
    # ─────────────────────────────────────
    def _cb_deteccion(self, msg: String):
        if self._secuencia_activa:
            self.get_logger().warn('Secuencia activa — ignorando detección')
            return

        tipo = msg.data.lower().strip()
        if tipo not in ('rack', 'rodillo'):
            self.get_logger().warn(f'Detección desconocida: {tipo}')
            return

        self._tipo_actual      = tipo
        self._alineado         = False
        self._secuencia_activa = True
        self.get_logger().info(f'Detección: {tipo} — iniciando secuencia')

        if tipo == 'rack':
            lib.spi_send_secuencia(
                CMD_RACK,
                T_RACK_SUBIR,
                T_RACK_SUBIR_POCO,
                T_RACK_DELAY,
                T_RACK_BAJAR_VISION,
                T_RACK_BAJAR_FINAL
            )
        else:  # rodillo
            lib.spi_send_secuencia(
                CMD_RODILLO,
                T_ROD_SUBIR,
                T_ROD_SUBIR_POCO,
                T_ROD_DELAY,
                T_ROD_BAJAR_VISION,
                T_ROD_BAJAR_FINAL
            )

        self.get_logger().info('Esperando ESPERAR_ALINEACION...')
        if not lib.esperar_estado(STATE_ESPERAR_ALINEACION, TIMEOUT):
            self.get_logger().error('TIMEOUT — FPGA no llegó a ESPERAR_ALINEACION')
            self._secuencia_activa = False
            return

        self.get_logger().info('FPGA listo — esperando alineación del robot...')

    # ─────────────────────────────────────
    # Alineación completada
    # ─────────────────────────────────────
    def _cb_alineacion(self, msg: Bool):
        if not self._secuencia_activa or not msg.data or self._alineado:
            return

        self._alineado = True
        self.get_logger().info('Robot alineado → mandando CMD_ALINEADO')
        lib.spi_send_command(CMD_ALINEADO)

        self.get_logger().info('Esperando ESPERAR_WAYPOINT...')
        if not lib.esperar_estado(STATE_ESPERAR_WAYPOINT, TIMEOUT):
            self.get_logger().error('TIMEOUT — FPGA no llegó a ESPERAR_WAYPOINT')
            self._secuencia_activa = False
            return

        self.get_logger().info('FPGA listo — esperando que robot llegue al destino...')

    # ─────────────────────────────────────
    # Waypoint alcanzado
    # ─────────────────────────────────────
    def _cb_waypoint_reached(self, _):
        if not self._secuencia_activa:
            return

        self.get_logger().info('Waypoint alcanzado → mandando CMD_WAYPOINT')
        lib.spi_send_command(CMD_WAYPOINT)

        self.get_logger().info('Esperando que FPGA deposite pallet...')
        if not lib.esperar_estado(STATE_IDLE, TIMEOUT):
            self.get_logger().error('TIMEOUT — FPGA no regresó a IDLE')
        else:
            self.get_logger().info(f'Secuencia {self._tipo_actual} completada ✓')

        self._secuencia_activa = False
        self._tipo_actual      = None
        self._alineado         = False


def main(args=None):
    rclpy.init(args=args)
    node = FpgaControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
