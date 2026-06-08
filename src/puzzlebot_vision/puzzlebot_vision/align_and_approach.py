#!/usr/bin/env python3
import os
import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
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
        # Odometría para el tramo final ciego (ver _cb_odom/_enter_blind más
        # abajo): decisión del usuario -- "es a pura cámara", sin LiDAR ni
        # sensores de contacto en el montacargas. La odometría (encoders vía
        # /odom, mismo tópico que usa mcl_node/mission_manager) NO es un
        # sensor externo de navegación: es la propiocepción del propio
        # robot, mide cuánto giraron SUS RUEDAS en respuesta a los comandos
        # que este mismo nodo manda -- categóricamente distinta de un
        # telémetro externo como el LiDAR. Estrategia acordada: "punto de
        # compromiso" confirmado por visión con estabilidad (ver
        # COMMIT_STABILITY_FRAMES) + tramo recto CORTO de longitud FIJA
        # (BLIND_APPROACH_DISTANCE, calibrada en campo) medido por
        # dead-reckoning desde ese punto -- mientras más corto el tramo,
        # menos importa la deriva acumulada de la odometría.
        self.create_subscription(Odometry, '/odom', self._cb_odom, 10)
        self._odom_xy = None         # (x, y) actual -- se actualiza en cada /odom
        self._blind_start_xy = None  # (x, y) capturado al entrar a BLIND -- ver _enter_blind

        self.publisher = self.create_publisher(CompressedImage, '/annotated_yolo_staged/compressed', 10)
        self.al_pub = self.create_publisher(CompressedImage, '/align_staged/compressed', 10)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Cierre del ciclo con mission_manager_node / fpga_controller_node.
        self.det_pub = self.create_publisher(Bool, '/pallet_detected', 10)
        self.qr_flag_pub = self.create_publisher(Bool, '/pallet_has_qr', 10)
        self.qr_content_pub = self.create_publisher(String, '/pallet_qr_content', 10)
        self.alineacion_pub = self.create_publisher(Bool, '/alineation/booleano', 10)

        # Objetivo de alineación/distancia con prioridad PALLET (ver
        # find_target): pallet+QR emparejados -> usa el bbox del PALLET
        # (grande y estable, mejor proxy de distancia y menos parpadeo
        # frame-a-frame); si el modelo no ve el pallet -> cae a usar el QR
        # solo, para no quedarse sin moverse (regresión que ya vivimos).
        self.class_ID = 3
        # El dataset de 'retrained.pt'/'lastlast.pt' etiquetó el MISMO QR
        # físico como dos clases distintas en frames distintos: 'qr' (id=4)
        # y 'qr-code' (id=5) — inconsistencia de etiquetado, no dos objetos
        # reales. Si solo aceptas una, pierdes detecciones al azar según
        # cuál "decida" usar el modelo en cada frame -> aceptar AMBAS.
        self.qr_class_IDs = (4, 5)

        # --- Phase 1 (ALIGNING): alineación por PULSOS, no P continuo ---
        #
        # Síntoma reportado en el robot real: la corrección giraba muy
        # despacio y, como el pasillo de acceso es corto, en algún punto el
        # motor "saltaba" (la velocidad angular comandada no se traduce de
        # forma lineal/predecible en giro real — zona muerta + fricción
        # estática) y el robot se desalineaba de nuevo sin alcanzar el
        # punto. Con P continuo, errores chicos generan consignas chicas
        # que no vencen esa zona muerta -> el controlador sigue subiendo la
        # consigna a ciegas hasta que el motor "suelta" de golpe: salto
        # descontrolado, y en un espacio corto no hay margen para que esa
        # oscilación converja.
        #
        # Solución robusta: en vez de un comando angular continuo, se manda
        # una RÁFAGA corta a velocidad fija (por encima de la zona muerta,
        # así el movimiento es predecible), luego el robot se DETIENE por
        # completo un instante (sin inercia, cámara capta un frame nítido
        # sin motion blur) y solo entonces se vuelve a medir el error real
        # para decidir la siguiente ráfaga. Cada corrección queda acotada
        # en magnitud — nunca "salta" más de lo esperado — y al re-medir
        # con datos frescos el sistema se autocorrige sin depender de un
        # modelo preciso del motor (justo lo que se necesita en un pasillo
        # angosto donde no hay margen para errores grandes).
        self.ALIGN_ERROR_THRESHOLD = 0.08
        self.ALIGN_PULSE_ANGULAR = 0.14       # rad/s de cada ráfaga
        self.ALIGN_PULSE_GAIN = 0.3           # duración de la ráfaga ∝ |error_x|: errores grandes -> pulsos más largos (converge rápido); chicos -> pulsos cortos (ajuste fino, resuelve el "muy despacio")
        # CORREGIDO (2026-06-08, "no se alinea, se queda quieto"): el piso
        # original (0.08s) resultó ser EL PROBLEMA, no la solución. Evidencia
        # de campo: tracking.py mueve al robot end-to-end con comandos
        # CONTINUOS de hasta solo 0.05 rad/s (más lento que estos 0.14 rad/s)
        # -- así que "vencer la zona muerta" no era cuestión de velocidad.
        # Era cuestión de TIEMPO: con fricción estática, el motor necesita un
        # instante para "soltarse" antes de empezar a girar de verdad; si la
        # ráfaga corta a los 80ms, vuelve a cero ANTES de que el motor llegue
        # a moverse -- toda la energía se va en vencer la fricción sin
        # producir giro neto, y el ciclo se repite sin converger nunca (el
        # síntoma exacto reportado: error_x clavado, nunca cambia). Subir el
        # piso le da al motor tiempo real de arrancar antes de cortar.
        # CALIBRAR EN CAMPO: si con 0.25s sigue sin moverse, seguir subiendo;
        # si se pasa de frenada (overshoot grande), ALIGN_OVERSHOOT_DAMPING
        # ya encoge el siguiente pulso geométricamente.
        self.ALIGN_PULSE_MIN_DURATION = 0.25  # s — piso: tiempo mínimo para que el motor REALMENTE empiece a girar, no solo "tenga suficiente velocidad"
        self.ALIGN_PULSE_MAX_DURATION = 0.45  # s — techo: ningún pulso gira "demasiado" de una sola vez en el pasillo angosto (subido en proporción al nuevo piso)
        self.ALIGN_SETTLE_DURATION = 0.30     # s de pausa total tras cada ráfaga, antes de volver a medir

        # Anti-oscilación: en campo, el "piso" ALIGN_PULSE_MIN_DURATION
        # resultó ser MÁS corrección de la que hacía falta para errores
        # chicos -> cada ráfaga se pasaba del centro (overshoot) y la
        # siguiente medición lanzaba la ráfaga contraria, que también se
        # pasaba: oscilación de amplitud constante que nunca entra en
        # ALIGN_ERROR_THRESHOLD (justo el síntoma reportado: "va un poco a
        # la derecha y depués a la izquierda pero no termina de centrarse").
        # Se detecta el overshoot con la firma más simple posible -- la
        # ráfaga que toca ahora invierte el signo de la anterior -- y
        # entonces se encoge la duración GEOMÉTRICAMENTE (bisección: cada
        # rebote, a la mitad de la mitad...). Esto sí puede perforar el
        # piso a propósito: una vez que se observó un rebote ya sabemos,
        # de forma empírica, que el motor responde de sobra, así que el
        # piso "para vencer la zona muerta" deja de hacer falta -- al
        # contrario, hace falta ir más fino que eso. Se restablece a 1.0
        # en cuanto una medición cae dentro del umbral SIN necesitar ráfaga
        # (ver el reinicio junto a `near_edge` más abajo), para que la
        # próxima realineación (p.ej. tras derivar durante el avance)
        # arranque otra vez a máxima potencia y no herede un paso ya
        # encogido por una corrección anterior mucho más grande.
        self.ALIGN_OVERSHOOT_DAMPING = 0.5

        # Estado de la máquina de pulsos (persiste entre frames):
        #   MEASURE -> decide si hace falta corregir y lanza el pulso
        #   PULSE   -> gira a velocidad fija durante una duración acotada
        #   SETTLE  -> se detiene por completo y deja que cámara/robot se asienten
        # (valores iniciales asignados por _reset_phase_state, ver más abajo)

        # Phase 2 (APPROACHING): drive forward using bbox area (image coverage) as a
        # distance proxy, with a small angular correction to stay centered.
        self.TARGET_AREA_RATIO = 0.35
        self.Kp_approach_linear = 0.6
        self.Kp_approach_correction = 0.1

        # --- Phase 3 (tramo final CIEGO, sin visión, sin LiDAR) ---
        #
        # A corta distancia el bbox crece hasta recortarse contra los
        # bordes del frame o salirse del FOV por completo — confiar en
        # visión hasta el final es frágil. Solución estándar en docking
        # visual: alinear con visión desde lejos, y el último tramo
        # recorrerlo con un mecanismo que no se degrada cerca.
        #
        # Restricción del usuario: "es a pura cámara" -- sin LiDAR ni
        # sensores de contacto en el montacargas. La única información
        # disponible para "cuánto falta" cuando la visión ya no ve nada es
        # la propia odometría del robot (ver _cb_odom) -- NO es un sensor
        # externo de navegación, es propiocepción (cuánto giraron las
        # ruedas en respuesta a los comandos que este nodo manda).
        #
        # Estrategia acordada con el usuario -- "punto de compromiso"
        # (commit point) confirmado por visión + tramo recto corto medido
        # por odometría:
        #   1) Detectar con la CÁMARA un punto repetible y cercano al
        #      punto real de carga -- el mismo par de señales que ya
        #      anticipaban la pérdida de FOV (área del bbox cerca de un
        #      umbral, o bbox tocando algún borde del frame) -- pero
        #      exigiendo que se sostenga COMMIT_STABILITY_FRAMES frames
        #      seguidos antes de comprometerse: un solo frame inflado por
        #      blur/oclusión no debe fijar el punto de partida del tramo
        #      ciego, de eso depende directamente dónde termina el robot.
        #   2) Desde ahí, recorrer BLIND_APPROACH_DISTANCE -- una distancia
        #      FIJA, CORTA y CALIBRADA EN CAMPO (parar al robot en el
        #      commit point, medir a mano cuánto falta hasta el punto real
        #      de carga, repetir varias veces y promediar) -- por dead-
        #      reckoning. Mientras más corto este tramo, menos importa la
        #      deriva acumulada de la odometría.
        # El usuario confirmó que "pasarse" (chocar contra el pallet/rack)
        # es el escenario peligroso y que el mecanismo levanta con error de
        # hasta 1cm -- por eso BLIND_APPROACH_DISTANCE debe calibrarse con
        # SESGO CONSERVADOR: el promedio medido en campo MENOS un colchón
        # de seguridad (mejor quedarse un poco corto -- tolerado por el
        # mecanismo -- que arriesgar overshoot).
        self.COMMIT_STABILITY_FRAMES = 5  # frames consecutivos en "zona de compromiso" antes de comprometerse -- filtra ruido de un solo frame

        self.BLIND_APPROACH_AREA_RATIO = 0.22  # área del pallet que marca la "zona de compromiso" -- debe ser MENOR que TARGET_AREA_RATIO (se evalúa antes de llegar al área objetivo por visión)
        self.BLIND_APPROACH_EDGE_MARGIN = 6    # px — bbox a esta distancia de cualquier borde = "a punto de salirse del FOV"
        self.BLIND_APPROACH_LINEAR = 0.05      # m/s durante el tramo final (recto, sin corrección — ya viene alineado de la fase anterior)

        # CALIBRAR EN CAMPO -- ver explicación arriba (commit point + sesgo
        # conservador). Valor placeholder, NO calibrado todavía:
        self.BLIND_APPROACH_DISTANCE = 0.05  # m -- "D_restante" medida y sesgada a la baja

        # Red de seguridad PURAMENTE temporal -- por si /odom deja de llegar
        # o el robot queda atascado (rueda patinando, obstáculo) sin que
        # _blind_traveled() avance: sin esto el robot seguiría empujando
        # indefinidamente. Derivado del tiempo nominal a BLIND_APPROACH_LINEAR
        # con margen amplio (2.5×) -- "definitivamente algo está mal", no un
        # cronómetro de operación normal.
        self.BLIND_APPROACH_TIMEOUT = (self.BLIND_APPROACH_DISTANCE / self.BLIND_APPROACH_LINEAR) * 2.5

        # 'VISION' (control normal por cámara) -> 'BLIND' (recto, a ciegas,
        # por distancia medida con odometría) -> 'DONE' (llegó, alto total,
        # AHÍ se publica /alineation/booleano=True -- nunca antes, ver
        # image_callback). Una vez que se entra a BLIND/DONE tiene prioridad
        # ABSOLUTA sobre la visión: el punto es justo dejar de confiar en lo
        # que la cámara (no) ve aquí.
        # (valor inicial asignado por _reset_phase_state, ver más abajo)

        # Compromiso con el objetivo: en cuanto se decodifica el QR una vez
        # (o se empareja geométricamente con el pallet), ya sabemos CON
        # CERTEZA que es el pallet correcto. Si DESPUÉS se pierde de vista
        # ya no tiene sentido frenar en seco a "buscar" -- vale más
        # comprometerse al tramo ciego que quedarse plantado a medio
        # pasillo (síntoma reportado: "se detiene y ya no avanza"). PERO
        # -- a diferencia de cuando había LiDAR -- comprometerse aquí solo
        # es seguro si YA estábamos cerca del punto de compromiso cuando se
        # perdió (ver _last_good_near_commit en image_callback): sin LiDAR
        # que mida la distancia real, BLIND_APPROACH_DISTANCE solo es válida
        # si el tramo ciego arranca aproximadamente desde donde se calibró.
        # (valor inicial asignado por _reset_phase_state, ver más abajo)

        # Ventana de gracia ante pérdida TOTAL estando LOCKED: un solo frame
        # sin detección (motion blur, glare, un mal frame de YOLO) no debería
        # bastar para comprometerse de forma IRREVERSIBLE al tramo ciego --
        # eso produciría un "ARRIVED" falso (el robot apenas avanza ~5cm y
        # se da por llegado) que dispararía a la FPGA a cargar en el lugar
        # equivocado. Se tolera la pérdida hasta TARGET_LOST_GRACE segundos:
        # durante esa ventana simplemente se PAUSA (sin comprometerse a
        # nada); si el objetivo reaparece, _lost_since se reinicia y se
        # sigue como si nada hubiera pasado. Solo si la pérdida persiste más
        # allá de la ventana se concluye que es real -> tramo ciego.
        self.TARGET_LOST_GRACE = 0.4  # s -- CALIBRAR: > un parpadeo típico de YOLO, < una pérdida real
        # (valor inicial de _lost_since asignado por _reset_phase_state, ver más abajo)

        self.MAX_LINEAR = 0.05
        # Debe ser >= ALIGN_PULSE_ANGULAR (0.14): ese valor se calibró en
        # campo como "la mínima velocidad que de verdad mueve al robot,
        # venciendo la zona muerta del motor" -- si este tope quedara por
        # debajo (como estaba: 0.1 < 0.14), np.clip recortaría CADA pulso a
        # un valor que vuelve a caer dentro de la zona muerta, reintroduciendo
        # justo el problema que la lógica de pulsos se diseñó para evitar
        # (el motor "se pega" por fricción estática y luego suelta de golpe
        # -- stick-slip: más giro real del comandado, y un salto en la
        # velocidad que reportan los encoders). Con 0.15 el pulso pasa sin
        # recortarse y el resto de usos de MAX_ANGULAR (corrección continua
        # en APPROACHING, línea ~542) ni lo notan -- ya quedan acotados muy
        # por debajo por Kp_approach_correction * error_x (máx. ~0.1).
        self.MAX_ANGULAR = 0.15

        # Todo el estado de fases/compromiso de arriba (_align_phase,
        # _approach_phase, _target_locked, _lost_since, ...) PERSISTE entre
        # frames a propósito -- es justo lo que hace funcionar las máquinas
        # de pulsos/aproximación/tramo ciego. Pero este mismo nodo se
        # reutiliza para las 4 áreas de pickup de mission_manager_node
        # (_PICKUP_AREAS = rodillos/rack1/rack2/rack3): sin resetearlo,
        # tras la primera llegada exitosa (_approach_phase='DONE') las
        # siguientes 3 reportarían "ARRIVED" fantasma de inmediato, sin
        # moverse, disparando carga de FPGA en el pallet equivocado. Se
        # inicializa aquí Y se vuelve a invocar en _reset_callback, disparado
        # por mission_manager_node justo antes de cada nuevo intento de
        # alineación (ver suscripción a /align_and_approach/reset abajo).
        # Arranque en frío: todavía no llegó ninguna confirmación de
        # mission_manager -- conservador (sin lock) hasta el primer reset
        # real (ver _reset_callback / Bool en vez de Empty más abajo).
        self._reset_phase_state(target_confirmed=False)

        self.reset_subscription = self.create_subscription(
            Bool,
            '/align_and_approach/reset',
            self._reset_callback,
            10
        )

        self.get_logger().info('Align-then-approach tracking started')

    def _reset_phase_state(self, target_confirmed: bool = False):
        """(Re)inicializa todo el estado persistente de fases/compromiso al
        valor de arranque de un intento de aproximación nuevo. Se llama una
        vez desde __init__ y de nuevo cada vez que mission_manager_node
        publica en /align_and_approach/reset (un nuevo pallet, posiblemente
        en otra área -- ver el comentario junto a self._reset_phase_state()
        en __init__ para el porqué).

        target_confirmed: viene del propio mensaje de reset (ver
        _reset_callback). mission_manager YA demostró durante el barrido
        -- con varias muestras y contenido de QR consistente, ver
        _evaluate_sweep_stop en mission_manager_node -- que este punto
        tiene un pallet con QR legible. Re-derivar esa misma certeza aquí
        cuadro a cuadro, en movimiento y de lejos (geometría de bbox
        emparejado o decodificación de QR, ver find_target/image_callback)
        es estrictamente más frágil: exige que el QR esté visible, bien
        resuelto y ambas cosas a la vez justo en la ventana transitoria
        antes de que salga del FOV. Si esa ventana se cierra sin lograrlo,
        _target_locked se queda en False para siempre y la primera pérdida
        total cae en la rama "nunca confirmado -> frenar sin red" (ver
        rama final de image_callback) -- exactamente el plantón reportado.
        Confiar en lo que mission_manager ya probó y arrancar LOCKED desde
        el frame 1 evita ese hueco por completo: el combo gracia/compromiso-
        a-ciegas queda activo de inmediato, sin depender de tener suerte."""
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

        # Gate de estabilidad del "punto de compromiso" (ver
        # COMMIT_STABILITY_FRAMES) -- cuenta frames CONSECUTIVOS en zona de
        # compromiso; cualquier frame que no cumpla lo resetea a 0.
        self._commit_streak = 0
        # Snapshot de si la ÚLTIMA detección válida (antes de perderse)
        # estaba ya en zona de compromiso -- decide si el disparo reactivo
        # por pérdida sostenida (ver rama final de image_callback) puede
        # comprometerse de forma segura a BLIND_APPROACH_DISTANCE (calibrada
        # para un tramo corto desde CERCA) o si el robot se perdió estando
        # todavía lejos, donde esa distancia se quedaría corta por mucho.
        self._last_good_near_commit = False

    def _reset_callback(self, msg: Bool):
        self.get_logger().info(
            f'Reset recibido (target_confirmed={msg.data}) -> reiniciando fases para nuevo intento de aproximación')
        self._reset_phase_state(target_confirmed=msg.data)

    def _cb_odom(self, msg: Odometry):
        """Guarda la posición (x, y) actual -- propiocepción del robot, NO
        un sensor externo de navegación (ver comentario junto a la
        suscripción en __init__). Es la única referencia de "cuánto avancé"
        disponible para el tramo ciego sin LiDAR ni sensores de contacto."""
        self._odom_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _blind_traveled(self) -> float:
        """Distancia recorrida (línea recta, Euclidiana) desde el instante
        en que se entró a BLIND (ver _enter_blind). Si /odom no ha llegado
        en ninguno de los dos extremos, devuelve 0.0 -- el tramo no avanza
        por distancia y queda solo el BLIND_APPROACH_TIMEOUT como red de
        seguridad (ver rama BLIND de image_callback)."""
        if self._odom_xy is None or self._blind_start_xy is None:
            return 0.0
        dx = self._odom_xy[0] - self._blind_start_xy[0]
        dy = self._odom_xy[1] - self._blind_start_xy[1]
        return math.hypot(dx, dy)

    def _enter_blind(self, now):
        """Punto único de entrada al tramo final. Los DOS disparos que existen
        -- anticipado por commit confirmado (target aún visible, ver rama
        `_commit_streak >= COMMIT_STABILITY_FRAMES`) y reactivo por pérdida
        sostenida estando LOCKED *Y* ya cerca del punto de compromiso (ver
        _last_good_near_commit en la rama final de image_callback) --
        convergen aquí. Captura la posición actual como origen para medir
        BLIND_APPROACH_DISTANCE por odometría (ver _blind_traveled);
        _blind_deadline queda solo como red de seguridad temporal."""
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
        """Elige el objetivo de alineación/distancia con prioridad PALLET:
        - pallet+QR emparejados (QR justo encima del pallet) -> objetivo =
          PALLET (bbox grande y estable, mejor proxy de distancia real,
          menos parpadeo); el QR se usa solo para decodificar contenido.
        - solo QR detectado -> fallback: objetivo = QR, para no perder al
          robot por completo si el modelo no ve el pallet en ese frame.
        - solo PALLET detectado (sin su QR en ESTE frame) Y self._target_locked
          ya es True (en algún frame anterior SÍ se vieron emparejados y se
          leyó su QR) -> fallback: objetivo = PALLET, sin qr_box. Es el caso
          típico al acercarse: el QR vive arriba del pallet y sale del FOV
          de la cámara antes que él -- pero como ya se confirmó una vez que
          ESTE pallet tiene QR, seguir usándolo como objetivo es seguro.
          Si NUNCA se vio su QR (self._target_locked sigue False), un
          pallet suelto no basta para comprometerse -- podría ser otro
          objeto pallet-forme en el encuadre -- así que NO se usa como
          objetivo todavía (se trata como "nada encontrado", ver más abajo).
        Acepta clase 4 ('qr') y 5 ('qr-code') como QR (ver self.qr_class_IDs).
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

            twist_msg = Twist()
            now = self.get_clock().now()
            target_box, qr_box, paired = self.find_target(results, h, w)

            detected = target_box is not None
            qr_text = None
            # Default explícito (no None): toda rama de abajo lo sobreescribe,
            # pero dejarlo en None invitaba a que `state == 'ARRIVED'` (línea
            # de alineacion_pub más abajo) "funcionara por accidente" en vez
            # de por diseño, y a que el overlay/registro de algunas ramas
            # (LOCKED-parpadeo, nunca-confirmado-perdido) quedaran sin texto.
            state = 'SEARCHING'

            # --- Tramo ciego y su conclusión: prioridad ABSOLUTA ---
            # Una vez que se decide entrar al tramo final, la visión queda
            # fuera del lazo de control por completo — es justo el punto:
            # a esta distancia ya no se puede confiar en lo que (no) ve la
            # cámara (FOV agotado / bbox recortado contra los bordes).
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
                    # Red de seguridad: o /odom nunca llegó (traveled=0.0
                    # todo este tramo) o el robot quedó atascado sin avanzar
                    # -- cualquiera de los dos es "algo está mal", mejor
                    # parar que seguir empujando más allá de lo razonable.
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
                # Se ve algo este frame -> rompe cualquier racha de pérdida
                # en curso (ver TARGET_LOST_GRACE): la próxima vez que se
                # pierda, la ventana de gracia arranca de cero.
                self._lost_since = None

                if paired:
                    # Pallet+QR detectados juntos con geometría consistente
                    # (ver find_target) -> ya sabemos que ESTE pallet tiene
                    # QR, sin necesidad de leerlo todavía. Detectar el bbox
                    # (lo que YOLO ya hace con buena confianza) es mucho más
                    # robusto que decodificarlo (pyzbar/cv2 necesitan buena
                    # resolución/foco/ángulo y pueden fallar aunque el bbox
                    # se vea clarísimo) -- exigir la decodificación dejaba
                    # _target_locked en False para siempre y el fallback de
                    # "pallet sin QR" de abajo nunca se activaba.
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

                # Punto rojo = centro del bbox objetivo (pallet si está
                # emparejado con su QR, o el QR solo en fallback).
                center_x = (xmin + xmax) // 2
                center_y = (ymin + ymax) // 2
                cv2.circle(align_frame, (center_x, center_y), 8, (0, 0, 255), -1)

                error_x = float(w_center - center_x) / (w / 2.0)
                area_ratio = float((xmax - xmin) * (ymax - ymin)) / float(w * h)
                near_edge = (xmin <= self.BLIND_APPROACH_EDGE_MARGIN
                             or xmax >= (w - self.BLIND_APPROACH_EDGE_MARGIN)
                             or ymax >= (h - self.BLIND_APPROACH_EDGE_MARGIN))

                if self._align_phase == 'MEASURE' and abs(error_x) <= self.ALIGN_ERROR_THRESHOLD:
                    # Quedó centrado sin necesitar ráfaga -> esta secuencia
                    # de alineación terminó limpia (convergió, con o sin
                    # rebotes en el camino). Olvida cualquier amortiguación
                    # acumulada -- la próxima vez que haga falta realinear
                    # (p.ej. tras derivar durante el avance) debe arrancar
                    # otra vez a máxima potencia, no heredar el último paso,
                    # ya encogido, de una corrección anterior y más grande.
                    self._align_pulse_scale = 1.0
                    self._align_pulse_sign = 0.0

                # La máquina de pulsos tiene prioridad sobre el resto de la
                # lógica de visión: un pulso o un asentamiento en curso
                # NUNCA se interrumpe a medias — interrumpirlo a medio giro
                # es justo lo que produce saltos impredecibles (el robot
                # queda "a la mitad" de un giro que no se sabe cuánto avanzó
                # realmente).
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
                    # Fase MEASURE con error real fuera de tolerancia ->
                    # mide fresco y lanza el siguiente pulso. Si el signo
                    # que toca ahora es CONTRARIO al de la última ráfaga,
                    # esa ráfaga se pasó del centro (overshoot): encoge el
                    # paso geométricamente con ALIGN_OVERSHOOT_DAMPING para
                    # converger en vez de rebotar siempre con la misma
                    # amplitud (ver el comentario junto a esa constante).
                    new_sign = 1.0 if error_x > 0 else -1.0
                    if self._align_pulse_sign != 0.0 and new_sign != self._align_pulse_sign:
                        self._align_pulse_scale *= self.ALIGN_OVERSHOOT_DAMPING
                    self._align_pulse_sign = new_sign

                    # Duración base proporcional al tamaño del error (grande
                    # = pulso largo, converge rápido; chico = pulso corto,
                    # ajuste fino) recortada a [piso, techo]; la amortiguación
                    # se aplica DESPUÉS del recorte y puede perforar el piso
                    # a propósito una vez detectado el rebote.
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
                    # BUG CORREGIDO (2026-06-08, "no corrige la alineación"):
                    # este bloque (detección de compromiso + control P de
                    # aproximación) vivía SUELTO, fuera del if/elif de la
                    # máquina de pulsos -- corría TODOS los frames sin
                    # importar la fase, y como se ejecuta DESPUÉS, su
                    # twist_msg/state pisaban siempre lo que acababan de
                    # decidir las ramas PULSE/SETTLE/lanzamiento de arriba.
                    # Resultado observado en campo: el log nunca mostraba
                    # "ALIGNING pulso..." (siempre "APPROACHING..."), y el
                    # robot solo corregía con el P continuo de
                    # Kp_approach_correction -- exactamente el control que
                    # el diseño de ráfagas reemplazó por no vencer la zona
                    # muerta del motor (ver comentario grande junto a
                    # ALIGN_PULSE_MIN_DURATION). Moverlo a este `else` lo
                    # vuelve mutuamente excluyente con la alineación: solo
                    # se avanza/aproxima cuando NINGUNA rama de arriba actuó,
                    # es decir, cuando MEASURE ya mide error dentro de
                    # ALIGN_ERROR_THRESHOLD (alineado de verdad).
                    #
                    # Ya alineado: ¿el blanco está a punto de salirse del FOV
                    # (área ya "cerca" o bbox tocando bordes)? Esa es la "zona
                    # de compromiso" -- la señal de que ya casi no hay visión
                    # útil. Pero NO nos comprometemos al primer frame que la
                    # cumple (un bbox inflado por blur/oclusión fijaría mal el
                    # origen del tramo ciego, y de ESE punto depende a dónde
                    # llega BLIND_APPROACH_DISTANCE): exigimos que se sostenga
                    # COMMIT_STABILITY_FRAMES frames seguidos.
                    near_commit = (area_ratio >= self.BLIND_APPROACH_AREA_RATIO or near_edge)
                    self._last_good_near_commit = near_commit
                    self._commit_streak = (self._commit_streak + 1) if near_commit else 0

                    if self._commit_streak >= self.COMMIT_STABILITY_FRAMES:
                        self._enter_blind(now)
                        state = 'APPROACHING (commit confirmado -> tramo final ciego por odometría)'
                        twist_msg.linear.x = self.BLIND_APPROACH_LINEAR
                        twist_msg.angular.z = 0.0

                    else:
                        # Sigue en aproximación normal por visión (control P por
                        # área/centrado) -- TARGET_AREA_RATIO funciona aquí solo
                        # como referencia de "qué tan rápido ir" (frena el avance
                        # conforme se acerca al área esperada de compromiso), NO
                        # como criterio de llegada -- esa decisión vive ahora
                        # ÚNICAMENTE en el tramo BLIND/odometría (ver arriba),
                        # para que exista un solo camino a 'DONE' y por tanto a
                        # /alineation/booleano=True.
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
                    # Ya leímos el QR de este pallet en algún frame anterior
                    # -> es el correcto, y para entonces estábamos lo
                    # bastante cerca/alineados como para decodificarlo.
                    # Perderlo ahora (oscilación que lo sacó del cuadro,
                    # blur, reflejo...) NO es motivo para clavar el freno a
                    # medio pasillo -- pero TAMPOCO para comprometerse de
                    # forma IRREVERSIBLE al primer frame malo: un parpadeo
                    # de un solo frame no debería bastar para declarar
                    # "ARRIVED" a 5cm de donde se perdió. Se mide hace
                    # cuánto empezó esta racha de pérdida (TARGET_LOST_GRACE)
                    # y se decide según eso.
                    if self._lost_since is None:
                        self._lost_since = now
                    lost_for = (now - self._lost_since).nanoseconds / 1e9

                    if lost_for < self.TARGET_LOST_GRACE:
                        # Ventana de gracia: probablemente solo un parpadeo
                        # (motion blur, reflejo, un mal frame de YOLO) --
                        # pausa sin comprometerse a nada. Si reaparece en el
                        # siguiente frame, _lost_since se reinicia arriba y
                        # seguimos exactamente donde íbamos, como si nada.
                        state = f'LOCKED (parpadeo {lost_for:.2f}s, esperando)'
                        twist_msg.linear.x = 0.0
                        twist_msg.angular.z = 0.0
                        cv2.putText(align_frame, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (255, 255, 0), 2)
                        self.get_logger().info(state)
                    elif self._last_good_near_commit:
                        # La pérdida ya duró más que un parpadeo -> es real.
                        # Y justo ANTES de perderlo ya estábamos en zona de
                        # compromiso (ver _last_good_near_commit) -- así que
                        # BLIND_APPROACH_DISTANCE (calibrada para un tramo
                        # CORTO desde ahí) sigue siendo válida: comprometerse
                        # es seguro y resuelve el plantón original ("se
                        # detiene y ya no avanza"). Mismo destino que el
                        # disparo anticipado (_enter_blind, ver rama
                        # `_commit_streak >= COMMIT_STABILITY_FRAMES` arriba).
                        self._enter_blind(now)
                        state = 'APPROACHING (target perdido cerca del compromiso -> tramo final ciego por odometría)'
                        twist_msg.linear.x = self.BLIND_APPROACH_LINEAR
                        twist_msg.angular.z = 0.0
                        cv2.putText(align_frame, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 165, 255), 2)
                        self.get_logger().warn(f'{state} -> linear={twist_msg.linear.x:.3f}')
                    else:
                        # Pérdida real, pero TODAVÍA LEJOS de la zona de
                        # compromiso -- a diferencia de cuando había LiDAR
                        # (medía la distancia real sin importar dónde
                        # arrancara el tramo ciego), sin él NO hay forma de
                        # saber cuánto falta desde aquí: comprometerse a
                        # BLIND_APPROACH_DISTANCE (calibrada para un tramo
                        # CORTO desde CERCA) dejaría al robot a medio camino,
                        # sin completar el acercamiento -- y el usuario
                        # confirmó que un mal cálculo aquí es justamente lo
                        # que NO se puede arriesgar ("pasarse" es lo
                        # peligroso; quedarse corto por mucho tampoco sirve).
                        # Más seguro: pausar y seguir intentando recuperar el
                        # objetivo (el lazo de control sigue corriendo cada
                        # frame) -- si nunca se recupera, el timeout propio de
                        # WAITING_ALIGNMENT en mission_manager (ver
                        # _cb_alineacion / [[project_mission_manager_plan]])
                        # aborta la misión con causa explícita en vez de
                        # quedarse colgada en silencio.
                        state = f'LOCKED (perdido lejos del punto de compromiso hace {lost_for:.1f}s -- pausando, esperando recuperar visión)'
                        twist_msg.linear.x = 0.0
                        twist_msg.angular.z = 0.0
                        cv2.putText(align_frame, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2)
                        self.get_logger().warn(state)
                else:
                    # Nunca se confirmó el objetivo (no se ha leído su QR
                    # todavía) -> no hay nada a qué comprometerse: frenar, y
                    # si reaparece, reiniciar la máquina de pulsos para medir
                    # error fresco en vez de continuar un pulso "fantasma"
                    # cuya duración restante ya no tiene sentido. (Si ya
                    # estamos en BLIND/DONE no se llega aquí: la visión ya
                    # no decide nada, por diseño.)
                    self._align_phase = 'MEASURE'
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.0
                    state = 'SEARCHING (sin objetivo confirmado, frenando)'
                    cv2.putText(align_frame, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)

            self.det_pub.publish(Bool(data=detected))
            self.qr_flag_pub.publish(Bool(data=qr_text is not None))
            if qr_text is not None:
                self.qr_content_pub.publish(String(data=qr_text))
            self.alineacion_pub.publish(Bool(data=(state == 'ARRIVED')))

            self.vel_pub.publish(twist_msg)
            self.publish(annotated_frame, align_frame)
        except Exception as e:
            # Fail-safe: un error de inferencia/decodificación no debe dejar
            # al robot repitiendo el último Twist publicado (que pudo ser
            # "avanza" o "gira") indefinidamente -- se manda velocidad cero
            # explícita antes de loguear, igual que cualquier otra rama que
            # decide frenar.
            self.vel_pub.publish(Twist())
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
