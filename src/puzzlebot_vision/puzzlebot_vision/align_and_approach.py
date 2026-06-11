#!/usr/bin/env python3
import os
import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String
from ament_index_python import get_package_share_directory
from pyzbar.pyzbar import decode as zbar_decode


class AlignAndApproach(Node):
    def __init__(self):
        super().__init__('align_and_approach_node')
        try:
            from ultralytics import YOLO
        except ImportError:
            self.get_logger().error('Ultralytics YOLO library not found')
            raise

        model_path = os.path.join(get_package_share_directory('puzzlebot_vision'), 'models', 'retrained.pt')
        self.model = YOLO(model_path)

        self.subscription = self.create_subscription(
            CompressedImage,
            '/video_source/compressed',
            self.image_callback,
            10
        )
        # /odom para el tramo ciego (dead-reckoning, ver _blind_traveled). QoS
        # debe ser BEST_EFFORT/VOLATILE como lo publica el robot -- con
        # RELIABLE rclpy nunca entrega los mensajes y _blind_traveled queda en 0.
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(Odometry, '/odom', self._cb_odom, odom_qos)
        self._odom_xy = None         # (x, y) actual
        self._blind_start_xy = None  # (x, y) al entrar a BLIND

        self.publisher = self.create_publisher(CompressedImage, '/annotated_yolo_staged/compressed', 10)
        self.al_pub = self.create_publisher(CompressedImage, '/align_staged/compressed', 10)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.det_pub = self.create_publisher(Bool, '/pallet_detected', 10)
        self.qr_flag_pub = self.create_publisher(Bool, '/pallet_has_qr', 10)
        self.qr_content_pub = self.create_publisher(String, '/pallet_qr_content', 10)
        self.alineacion_pub = self.create_publisher(Bool, '/alineation/booleano', 10)

        self.class_ID = 3  # 'pallet': prioridad sobre QR en find_target (bbox más grande/estable)
        self.qr_class_IDs = (4, 5)  # 'qr' y 'qr-code': mismo QR físico, etiquetado inconsistente en el dataset

        # Phase 1 (ALIGNING) por RÁFAGAS, no P continuo: el motor tiene zona
        # muerta + fricción estática, así que un P continuo oscila sin
        # converger en un pasillo corto. En su lugar: ráfaga corta a
        # velocidad fija, parada total (frame nítido), remedir, repetir.
        self.ALIGN_ERROR_THRESHOLD = 0.08
        self.ALIGN_PULSE_ANGULAR = 0.14       # rad/s de cada ráfaga
        self.ALIGN_PULSE_GAIN = 0.3           # duración ∝ |error_x|
        self.ALIGN_PULSE_MIN_DURATION = 0.25  # s -- piso: lo que tarda el motor en realmente empezar a girar
        self.ALIGN_PULSE_MAX_DURATION = 0.45  # s -- techo: ningún pulso gira de más en el pasillo angosto
        self.ALIGN_SETTLE_DURATION = 0.30     # s de pausa tras cada ráfaga antes de remedir

        # Anti-oscilación: el piso de duración sobre-corrige errores chicos
        # (overshoot en zigzag). Si el signo de la ráfaga se invierte respecto
        # a la anterior, se encoge la duración geométricamente; se restablece
        # a 1.0 al converger, para que la próxima corrección arranque a tope.
        self.ALIGN_OVERSHOOT_DAMPING = 0.5

        # Máquina de pulsos (estado persiste entre frames, ver _reset_phase_state):
        #   MEASURE -> mide error y decide si lanza pulso
        #   PULSE   -> gira a velocidad fija una duración acotada
        #   SETTLE  -> se detiene para que cámara/robot se asienten

        # Phase 2 (APPROACHING): avanza usando el área del bbox como proxy de
        # distancia, con corrección angular leve para mantenerse centrado.
        self.TARGET_AREA_RATIO = 0.35
        self.Kp_approach_linear = 0.6
        self.Kp_approach_correction = 0.1

        # Phase 3 (tramo final CIEGO): de cerca el bbox
        # se recorta contra los bordes o sale del FOV, así que el tramo final
        # se recorre por dead-reckoning desde un "punto de compromiso"
        # detectado por visión (área/bordes, sostenido COMMIT_STABILITY_FRAMES
        # frames para filtrar ruido) hasta BLIND_APPROACH_DISTANCE -- una
        # distancia FIJA y CORTA calibrada en campo (a menor tramo, menos
        # importa la deriva de odometría). "Pasarse" es lo peligroso, así que
        # se calibra con sesgo conservador (promedio medido menos colchón).
        self.COMMIT_STABILITY_FRAMES = 5

        self.BLIND_APPROACH_AREA_RATIO = 0.22  # área que marca "zona de compromiso" (< TARGET_AREA_RATIO)
        self.BLIND_APPROACH_EDGE_MARGIN = 6    # px -- bbox a esta distancia de un borde = a punto de salir del FOV
        self.BLIND_APPROACH_LINEAR = 0.05      # m/s en el tramo final (recto, sin corrección)

        self.BLIND_APPROACH_DISTANCE = 0.05  # m -- CALIBRAR EN CAMPO, placeholder no calibrado

        # Red de seguridad temporal si /odom no llega o el robot se atasca.
        self.BLIND_APPROACH_TIMEOUT = (self.BLIND_APPROACH_DISTANCE / self.BLIND_APPROACH_LINEAR) * 2.5

        # Red de seguridad: si ya está alineado (error_x dentro de umbral) pero
        # near_commit (área/borde) nunca se cumple -- o si tras alinearse se
        # pierde el target sin haber llegado a near_commit -- el robot se
        # quedaría congelado para siempre sin publicar ARRIVED. Para entonces
        # los forks ya están arriba (SEND_DETECCION ya se mandó), así que tras
        # este tiempo se fuerza el tramo final ciego de todos modos: es más
        # seguro que abortar la misión por WAITING_ALIGNMENT con los forks
        # levantados.
        self.STUCK_NEAR_TIMEOUT = 4.0  # s

        # Fases de _approach_phase: 'VISION' -> 'BLIND' (odometría) -> 'DONE'
        # (publica /alineation/booleano=True). BLIND/DONE tienen prioridad
        # absoluta -- la visión deja de decidir nada. (init en _reset_phase_state)

        # _target_locked: una vez confirmado el QR, perderlo de vista no debe
        # frenar la misión -- comprometerse al tramo ciego es más seguro que
        # quedarse plantado, siempre que ya estuviéramos cerca del punto de
        # compromiso (ver _last_good_near_commit). (init en _reset_phase_state)

        # TARGET_LOST_GRACE: un solo frame sin detección (blur, glare) no debe
        # bastar para comprometerse al tramo ciego (produciría un "ARRIVED"
        # falso). Se tolera la pérdida hasta este tiempo pausando sin
        # comprometerse. CALIBRAR: > un parpadeo típico de YOLO, < pérdida real.
        self.TARGET_LOST_GRACE = 0.4  # s

        self.MAX_LINEAR = 0.05
        # Debe ser >= ALIGN_PULSE_ANGULAR (0.14): si quedara por debajo,
        # np.clip recortaría cada pulso de vuelta a la zona muerta del motor,
        # reintroduciendo el stick-slip que las ráfagas evitan.
        self.MAX_ANGULAR = 0.15

        # El nodo se reutiliza para las 4 áreas de pickup de mission_manager
        # -- sin resetear, tras la primera llegada las siguientes reportarían
        # "ARRIVED" fantasma. Se reinvoca desde _reset_callback en cada intento.
        self._reset_phase_state(target_confirmed=False)

        self.reset_subscription = self.create_subscription(
            Bool,
            '/align_and_approach/reset',
            self._reset_callback,
            10
        )

        # Activo SOLO durante la aproximación final (mission_manager lo pone
        # en True justo antes de SEND_DETECCION y en False al confirmar
        # /alineation/booleano). Mientras está en False (barrido, navegación,
        # etc.) la percepción (YOLO + decode QR + /pallet_detected,
        # /pallet_has_qr, /pallet_qr_content) sigue corriendo -- el barrido
        # de mission_manager depende de ella -- pero la máquina de fases de
        # alineación/aproximación NO avanza y este nodo NO publica /cmd_vel,
        # para no competir con los giros del barrido (eso hacía que el
        # robot se pasara del pallet correcto antes de poder muestrearlo).
        self._active = False
        self.create_subscription(
            Bool, '/align_and_approach/active', self._active_callback, 10)

        self.get_logger().info('Align-then-approach tracking started')

    def _active_callback(self, msg: Bool):
        if self._active and not msg.data:
            # Cede /cmd_vel: publica un Twist() para no dejar el último
            # comando "pegado" si justo en este instante no era cero.
            self.vel_pub.publish(Twist())
        self._active = msg.data

    def _reset_phase_state(self, target_confirmed: bool = False):
        """Reinicia el estado de fases para un nuevo intento de aproximación.
        target_confirmed viene de mission_manager, que ya verificó el QR
        durante el barrido -- confiar en eso evita re-confirmarlo en movimiento."""
        now = self.get_clock().now()

        self._align_phase = 'MEASURE'
        self._align_phase_deadline = now
        self._align_pulse_sign = 0.0
        self._align_pulse_scale = 1.0

        self._approach_phase = 'VISION'
        self._blind_deadline = now
        self._blind_start_xy = None

        self._target_locked = bool(target_confirmed)
        self._lost_since = None
        self._locked_center = None  # (cx, cy) del último target bloqueado, ver find_target

        self._commit_streak = 0           # frames consecutivos en zona de compromiso
        self._last_good_near_commit = False  # si la última detección antes de perderse ya estaba ahí
        self._near_aligned_since = None   # timestamp desde que error_x entró al umbral, ver STUCK_NEAR_TIMEOUT

    def _reset_callback(self, msg: Bool):
        self.get_logger().info(
            f'Reset recibido (target_confirmed={msg.data}) -> reiniciando fases para nuevo intento de aproximación')
        self._reset_phase_state(target_confirmed=msg.data)

    def _cb_odom(self, msg: Odometry):
        self._odom_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _blind_traveled(self) -> float:
        """Distancia recta recorrida desde que entró a BLIND (0.0 si /odom no ha llegado)."""
        if self._odom_xy is None or self._blind_start_xy is None:
            return 0.0
        dx = self._odom_xy[0] - self._blind_start_xy[0]
        dy = self._odom_xy[1] - self._blind_start_xy[1]
        return math.hypot(dx, dy)

    def _enter_blind(self, now):
        """Entra al tramo final: captura el origen para medir por odometría."""
        self._approach_phase = 'BLIND'
        self._blind_start_xy = self._odom_xy
        if self._blind_start_xy is None:
            self.get_logger().warn(
                'Entrando a BLIND sin /odom recibido aún -- _blind_traveled() '
                'quedará en 0.0 hasta que llegue; el timeout de seguridad es '
                'el único respaldo mientras tanto')
        self._blind_deadline = now + Duration(seconds=self.BLIND_APPROACH_TIMEOUT)

    def decode(self, roi):
        try:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            qr_codes = zbar_decode(gray_roi)
            if qr_codes:
                return qr_codes[0].data.decode('utf-8')
            return None
        except ImportError:
            detector = cv2.QRCodeDetector()
            data, bbox, straight_qrcode = detector.detectAndDecode(roi)
            if bbox is not None and data:
                return data
            return None
        except Exception:
            return None

    def find_target(self, results, h, w):
        """Prioridad PALLET: pallet+QR emparejados -> objetivo=pallet (bbox
        más estable); solo QR -> fallback=QR; solo pallet ya con _target_locked
        -> fallback=pallet MÁS CERCANO a _locked_center (el QR suele salir
        del FOV primero al acercarse, y puede haber otros pallets sin QR en
        cuadro -- sin esto, el robot podía saltar a cualquiera de ellos e
        ignorar el que ya había identificado por QR).
        Retorna (target_box, qr_box, paired) o (None, None, False)."""
        boxes_pallet, boxes_qr = [], []
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy().astype(int)
            coords = (max(0, xmin), max(0, ymin), min(w, xmax), min(h, ymax))
            if cls_id == self.class_ID:
                boxes_pallet.append(coords)
            elif cls_id in self.qr_class_IDs:
                boxes_qr.append(coords)

        for pallet in boxes_pallet:
            xmin_p, ymin_p, xmax_p, ymax_p = pallet
            center_x_p = (xmin_p + xmax_p) // 2
            for qr in boxes_qr:
                xmin_q, ymin_q, xmax_q, ymax_q = qr
                center_x_q = (xmin_q + xmax_q) // 2
                # Geometría esperada: el QR va arriba y solapado horizontalmente con el pallet
                if ymax_q <= (ymin_p + 20) and (xmin_p <= center_x_q <= xmax_p or xmin_q <= center_x_p <= xmax_q):
                    return pallet, qr, True

        if boxes_qr:
            return boxes_qr[0], boxes_qr[0], False
        if boxes_pallet and self._target_locked:
            if self._locked_center is not None and len(boxes_pallet) > 1:
                lcx, lcy = self._locked_center
                def dist2(p):
                    cx = (p[0] + p[2]) / 2.0
                    cy = (p[1] + p[3]) / 2.0
                    return (cx - lcx) ** 2 + (cy - lcy) ** 2
                return min(boxes_pallet, key=dist2), None, False
            return boxes_pallet[0], None, False
        return None, None, False

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                return

            h, w, _ = cv_image.shape
            h_center, w_center = h // 2, w // 2

            results = self.model(cv_image, device='cpu', verbose=False)
            annotated_frame = results[0].plot()
            align_frame = annotated_frame.copy()
            cv2.line(align_frame, (w_center, 0), (w_center, h), (0, 255, 0), 2)

            target_box, qr_box, paired = self.find_target(results, h, w)
            detected = target_box is not None
            qr_text = None

            if paired:
                # Pallet+QR emparejados por geometría -> ya sabemos que tiene
                # QR sin necesidad de decodificarlo (más robusto que
                # pyzbar/cv2, que exigen buena resolución/foco/ángulo).
                if not self._target_locked:
                    self.get_logger().info(
                        'Target LOCKED (pallet+QR emparejados por geometría)')
                self._target_locked = True

            if qr_box is not None:
                qxmin, qymin, qxmax, qymax = qr_box
                if qymax > qymin and qxmax > qxmin:
                    qr_text = self.decode(cv_image[qymin:qymax, qxmin:qxmax])
                    if qr_text is not None:
                        self.get_logger().info(f'QR: {qr_text}')
                        self._target_locked = True

            # Percepción: estos topics los consume mission_manager durante el
            # barrido (con self._active=False) para decidir a qué pN ir, así
            # que se publican SIEMPRE, sin importar self._active.
            self.det_pub.publish(Bool(data=detected))
            self.qr_flag_pub.publish(Bool(data=qr_text is not None))
            if qr_text is not None:
                self.qr_content_pub.publish(String(data=qr_text))

            if not self._active:
                # Fuera de la aproximación final (barrido, navegación, etc.):
                # no avanzar la máquina de fases ni tocar /cmd_vel -- ver
                # _active_callback / comentario en __init__.
                self.alineacion_pub.publish(Bool(data=False))
                self.publish(annotated_frame, align_frame)
                return

            twist_msg = Twist()
            now = self.get_clock().now()
            state = 'SEARCHING'

            if self._approach_phase == 'BLIND':
                state = 'APPROACHING (tramo final, ciego por odometría)'
                twist_msg.linear.x = self.BLIND_APPROACH_LINEAR
                twist_msg.angular.z = 0.0
                traveled = self._blind_traveled()
                if traveled >= self.BLIND_APPROACH_DISTANCE:
                    self._approach_phase = 'DONE'
                    twist_msg.linear.x = 0.0
                    self.get_logger().info(
                        f'{state} -> avanzado={traveled:.3f} m / objetivo={self.BLIND_APPROACH_DISTANCE:.3f} m '
                        '-- LLEGADA confirmada por odometría, deteniendo')
                elif now >= self._blind_deadline:
                    self.get_logger().warn(
                        f'{state}: timeout sin completar la distancia objetivo '
                        f'(avanzado={traveled:.3f} m / objetivo={self.BLIND_APPROACH_DISTANCE:.3f} m) '
                        '-> deteniendo por seguridad')
                    self._approach_phase = 'DONE'
                    twist_msg.linear.x = 0.0
                else:
                    self.get_logger().info(
                        f'{state} -> avanzado={traveled:.3f} m / objetivo={self.BLIND_APPROACH_DISTANCE:.3f} m '
                        f'(linear={twist_msg.linear.x:.3f})')

            elif self._approach_phase == 'DONE':
                state = 'ARRIVED'
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0

            elif target_box is not None:
                xmin, ymin, xmax, ymax = target_box
                self._lost_since = None

                center_x = (xmin + xmax) // 2
                center_y = (ymin + ymax) // 2
                cv2.circle(align_frame, (center_x, center_y), 8, (0, 0, 255), -1)

                if self._target_locked:
                    self._locked_center = (center_x, center_y)

                error_x = float(w_center - center_x) / (w / 2.0)
                area_ratio = float((xmax - xmin) * (ymax - ymin)) / float(w * h)
                near_edge = (xmin <= self.BLIND_APPROACH_EDGE_MARGIN
                             or xmax >= (w - self.BLIND_APPROACH_EDGE_MARGIN)
                             or ymax >= (h - self.BLIND_APPROACH_EDGE_MARGIN))

                if self._align_phase == 'MEASURE' and abs(error_x) <= self.ALIGN_ERROR_THRESHOLD:
                    # Convergió sin ráfaga -> resetea la amortiguación acumulada
                    self._align_pulse_scale = 1.0
                    self._align_pulse_sign = 0.0

                if self._align_phase == 'PULSE':
                    state = f'ALIGNING pulso({"+" if self._align_pulse_sign > 0 else "-"})'
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = float(np.clip(
                        self._align_pulse_sign * self.ALIGN_PULSE_ANGULAR, -self.MAX_ANGULAR, self.MAX_ANGULAR))
                    if now >= self._align_phase_deadline:
                        self._align_phase = 'SETTLE'
                        self._align_phase_deadline = now + Duration(seconds=self.ALIGN_SETTLE_DURATION)

                elif self._align_phase == 'SETTLE':
                    state = 'ALIGNING asentando'
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.0
                    if now >= self._align_phase_deadline:
                        self._align_phase = 'MEASURE'

                elif abs(error_x) > self.ALIGN_ERROR_THRESHOLD:
                    # Todavía corrigiendo -> aún no se considera "alineado"
                    self._near_aligned_since = None

                    # Signo invertido respecto a la última ráfaga -> overshoot, amortigua
                    new_sign = 1.0 if error_x > 0 else -1.0
                    if self._align_pulse_sign != 0.0 and new_sign != self._align_pulse_sign:
                        self._align_pulse_scale *= self.ALIGN_OVERSHOOT_DAMPING
                    self._align_pulse_sign = new_sign

                    base_duration = float(np.clip(
                        abs(error_x) * self.ALIGN_PULSE_GAIN,
                        self.ALIGN_PULSE_MIN_DURATION, self.ALIGN_PULSE_MAX_DURATION))
                    pulse_duration = base_duration * self._align_pulse_scale

                    self._align_phase = 'PULSE'
                    self._align_phase_deadline = now + Duration(seconds=pulse_duration)
                    state = f'ALIGNING pulso({"+" if self._align_pulse_sign > 0 else "-"} x{self._align_pulse_scale:.2f})'
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = float(np.clip(
                        self._align_pulse_sign * self.ALIGN_PULSE_ANGULAR, -self.MAX_ANGULAR, self.MAX_ANGULAR))

                else:
                    # Ya alineado (mutuamente excluyente con la máquina de
                    # pulsos -- debe vivir aquí o pisa su twist_msg). Zona de
                    # compromiso = área cerca del umbral o bbox tocando bordes,
                    # sostenida COMMIT_STABILITY_FRAMES frames para filtrar ruido.
                    near_commit = (area_ratio >= self.BLIND_APPROACH_AREA_RATIO or near_edge)
                    self._last_good_near_commit = near_commit
                    self._commit_streak = (self._commit_streak + 1) if near_commit else 0

                    if self._near_aligned_since is None:
                        self._near_aligned_since = now
                    stuck_for = (now - self._near_aligned_since).nanoseconds / 1e9

                    if self._commit_streak >= self.COMMIT_STABILITY_FRAMES:
                        self._enter_blind(now)
                        state = 'APPROACHING (commit confirmado -> tramo final ciego por odometría)'
                        twist_msg.linear.x = self.BLIND_APPROACH_LINEAR
                        twist_msg.angular.z = 0.0

                    elif stuck_for >= self.STUCK_NEAR_TIMEOUT:
                        self._enter_blind(now)
                        state = (f'APPROACHING (alineado hace {stuck_for:.1f}s sin confirmar commit '
                                 '-> forzando tramo final ciego)')
                        twist_msg.linear.x = self.BLIND_APPROACH_LINEAR
                        twist_msg.angular.z = 0.0
                        self.get_logger().warn(state)

                    else:
                        # Control P por área/centrado; TARGET_AREA_RATIO solo
                        # regula velocidad, no es criterio de llegada (eso
                        # vive únicamente en BLIND/odometría).
                        state = ('APPROACHING' if not near_commit else
                                 f'APPROACHING (confirmando commit {self._commit_streak}/{self.COMMIT_STABILITY_FRAMES})')
                        area_error = self.TARGET_AREA_RATIO - area_ratio
                        twist_msg.linear.x = float(np.clip(
                            area_error * self.Kp_approach_linear, 0.0, self.MAX_LINEAR))
                        twist_msg.angular.z = float(np.clip(
                            error_x * self.Kp_approach_correction, -self.MAX_ANGULAR, self.MAX_ANGULAR))

                if paired:
                    target_desc = 'pallet+QR'
                elif qr_box is not None:
                    target_desc = 'QR fallback'
                else:
                    target_desc = 'pallet (sin QR este frame)'
                label = f'{state} ({target_desc})'
                cv2.putText(align_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
                self.get_logger().info(
                    f'{label} | error_x={error_x:.2f} area_ratio={area_ratio:.2f} '
                    f'-> linear={twist_msg.linear.x:.3f} angular={twist_msg.angular.z:.3f}')
            else:
                self.get_logger().info(
                    f'Pallet (class {self.class_ID}) / QR (clases {self.qr_class_IDs}) not found.')

                if self._target_locked:
                    if self._lost_since is None:
                        self._lost_since = now
                    lost_for = (now - self._lost_since).nanoseconds / 1e9

                    if lost_for < self.TARGET_LOST_GRACE:
                        # Probablemente solo un parpadeo: pausa sin comprometerse
                        state = f'LOCKED (parpadeo {lost_for:.2f}s, esperando)'
                        twist_msg.linear.x = 0.0
                        twist_msg.angular.z = 0.0
                        cv2.putText(align_frame, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (255, 255, 0), 2)
                        self.get_logger().info(state)
                    elif self._last_good_near_commit:
                        # Pérdida real, pero ya estábamos en zona de compromiso
                        # -> BLIND_APPROACH_DISTANCE sigue siendo válida desde ahí
                        self._enter_blind(now)
                        state = 'APPROACHING (target perdido cerca del compromiso -> tramo final ciego por odometría)'
                        twist_msg.linear.x = self.BLIND_APPROACH_LINEAR
                        twist_msg.angular.z = 0.0
                        cv2.putText(align_frame, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 165, 255), 2)
                        self.get_logger().warn(f'{state} -> linear={twist_msg.linear.x:.3f}')
                    elif (self._near_aligned_since is not None
                          and (now - self._near_aligned_since).nanoseconds / 1e9 >= self.STUCK_NEAR_TIMEOUT):
                        # Ya estaba alineado antes de perderse y lleva
                        # STUCK_NEAR_TIMEOUT sin recuperar visión ni confirmar
                        # commit -> forzar tramo final ciego (ver STUCK_NEAR_TIMEOUT).
                        self._enter_blind(now)
                        state = 'APPROACHING (perdido tras alinear, timeout -> tramo final ciego)'
                        twist_msg.linear.x = self.BLIND_APPROACH_LINEAR
                        twist_msg.angular.z = 0.0
                        cv2.putText(align_frame, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 165, 255), 2)
                        self.get_logger().warn(state)
                    else:
                        # Lejos del punto de compromiso: sin LiDAR no se sabe
                        # cuánto falta, mejor pausar y esperar a recuperar visión
                        # (si persiste, WAITING_ALIGNMENT en mission_manager aborta).
                        state = f'LOCKED (perdido lejos del punto de compromiso hace {lost_for:.1f}s -- pausando, esperando recuperar visión)'
                        twist_msg.linear.x = 0.0
                        twist_msg.angular.z = 0.0
                        cv2.putText(align_frame, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2)
                        self.get_logger().warn(state)
                else:
                    # Nunca confirmado -> frenar y reiniciar la máquina de pulsos
                    self._align_phase = 'MEASURE'
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.0
                    state = 'SEARCHING (sin objetivo confirmado, frenando)'
                    cv2.putText(align_frame, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)

            self.alineacion_pub.publish(Bool(data=(state == 'ARRIVED')))

            self.vel_pub.publish(twist_msg)
            self.publish(annotated_frame, align_frame)
        except Exception as e:
            self.vel_pub.publish(Twist())  # fail-safe: no repetir el último Twist si falla la inferencia
            self.get_logger().error(f'Inference failed: {e}')

    def publish(self, annotated, alignment):
        _, buf_ann = cv2.imencode('.jpg', annotated)
        _, buf_al = cv2.imencode('.jpg', alignment)

        ann_msg = CompressedImage()
        ann_msg.header.stamp = self.get_clock().now().to_msg()
        ann_msg.format = 'jpeg'
        ann_msg.data = buf_ann.tobytes()

        al_msg = CompressedImage()
        al_msg.header.stamp = self.get_clock().now().to_msg()
        al_msg.format = 'jpeg'
        al_msg.data = buf_al.tobytes()

        self.publisher.publish(ann_msg)
        self.al_pub.publish(al_msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = AlignAndApproach()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
