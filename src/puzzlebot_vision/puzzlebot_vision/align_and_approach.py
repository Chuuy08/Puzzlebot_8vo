#!/usr/bin/env python3
import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
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
        self.ALIGN_PULSE_ANGULAR = 0.18       # rad/s de cada ráfaga — CALIBRAR: la mínima velocidad que de verdad mueve al robot (vence zona muerta) sin pasarse
        self.ALIGN_PULSE_GAIN = 0.6           # duración de la ráfaga ∝ |error_x|: errores grandes -> pulsos más largos (converge rápido); chicos -> pulsos cortos (ajuste fino, resuelve el "muy despacio")
        self.ALIGN_PULSE_MIN_DURATION = 0.08  # s — piso: nunca tan corto que ni alcance a vencer la inercia de arranque
        self.ALIGN_PULSE_MAX_DURATION = 0.35  # s — techo: ningún pulso gira "demasiado" de una sola vez en el pasillo angosto
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
        self._align_phase = 'MEASURE'
        self._align_phase_deadline = self.get_clock().now()
        self._align_pulse_sign = 0.0
        self._align_pulse_scale = 1.0

        # Phase 2 (APPROACHING): drive forward using bbox area (image coverage) as a
        # distance proxy, with a small angular correction to stay centered.
        self.TARGET_AREA_RATIO = 0.35
        self.Kp_approach_linear = 0.6
        self.Kp_approach_correction = 0.1

        # --- Phase 3 (tramo final CIEGO, sin visión) ---
        #
        # A corta distancia el bbox crece hasta recortarse contra los
        # bordes del frame o salirse del FOV por completo — confiar en
        # visión hasta el final es frágil (justo lo que preguntaste: "va a
        # perder visibilidad para llegar y terminar de corregir"). Solución
        # estándar en docking visual: alinear con visión desde lejos, y el
        # último tramo recorrerlo "de memoria" (open-loop) — para entonces
        # el robot YA viene bien alineado de la fase anterior, así que basta
        # avanzar derecho un tiempo fijo y calibrado, y detenerse.
        #
        # El disparo NO espera a "perder la detección" (ya sería tarde — ni
        # se sabría a qué distancia real está); se anticipa con dos señales
        # que aparecen ANTES de perder de vista al blanco:
        #   1) el área ya alcanzó un umbral "cerca" (más chico que
        #      TARGET_AREA_RATIO, calculado para dispararse con el bbox aún
        #      completamente visible), o
        #   2) el bbox ya está tocando algún borde del frame.
        self.BLIND_APPROACH_AREA_RATIO = 0.22  # área que dispara el tramo ciego — debe ser MENOR que TARGET_AREA_RATIO (se dispara antes de "llegar" según visión)
        self.BLIND_APPROACH_EDGE_MARGIN = 6    # px — bbox a esta distancia de cualquier borde = "a punto de salirse del FOV"
        self.BLIND_APPROACH_DURATION = 1.2     # s — CALIBRAR EN CAMPO: cuánto falta por recorrer cuando se dispara, a BLIND_APPROACH_LINEAR
        self.BLIND_APPROACH_LINEAR = 0.04      # m/s durante el tramo ciego (recto, sin corrección — ya viene alineado)

        # 'VISION' (control normal por cámara) -> 'BLIND' (recto, a ciegas,
        # por tiempo fijo) -> 'DONE' (llegó, alto total). Una vez que se
        # entra a BLIND/DONE tiene prioridad ABSOLUTA sobre la visión: el
        # punto es justo dejar de confiar en lo que la cámara (no) ve aquí.
        self._approach_phase = 'VISION'
        self._blind_deadline = self.get_clock().now()

        # Compromiso con el objetivo: en cuanto se decodifica el QR una vez,
        # ya sabemos CON CERTEZA que es el pallet correcto -- y que para
        # entonces estábamos lo bastante cerca/alineados como para leerlo.
        # Si DESPUÉS se pierde de vista (oscilación que lo sacó del cuadro,
        # blur, reflejo, lo que sea) ya no tiene sentido frenar en seco
        # a "buscar": vale más comprometerse al tramo ciego (mismo mecanismo
        # que BLIND_APPROACH_*, ver rama `else` de image_callback) que
        # quedarse plantado a medio pasillo (síntoma reportado: "se detiene
        # y ya no avanza").
        self._target_locked = False

        self.MAX_LINEAR = 0.05
        self.MAX_ANGULAR = 0.4

        self.get_logger().info('Align-then-approach tracking started')

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
            state = None

            # --- Tramo ciego y su conclusión: prioridad ABSOLUTA ---
            # Una vez que se decide entrar al tramo final, la visión queda
            # fuera del lazo de control por completo — es justo el punto:
            # a esta distancia ya no se puede confiar en lo que (no) ve la
            # cámara (FOV agotado / bbox recortado contra los bordes).
            if self._approach_phase == 'BLIND':
                state = 'APPROACHING (tramo ciego)'
                twist_msg.linear.x = self.BLIND_APPROACH_LINEAR
                twist_msg.angular.z = 0.0
                if now >= self._blind_deadline:
                    self._approach_phase = 'DONE'
                self.get_logger().info(f'{state} -> linear={twist_msg.linear.x:.3f}')

            elif self._approach_phase == 'DONE':
                state = 'ARRIVED'
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0

            elif target_box is not None:
                xmin, ymin, xmax, ymax = target_box

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

                # Ya alineado: ¿el blanco está a punto de salirse del FOV
                # (área ya "cerca" o bbox tocando bordes)? -> última recta a
                # ciegas, en vez de esperar a perder la detección de verdad
                # (para entonces ya no sabríamos ni a qué distancia está).
                elif area_ratio >= self.BLIND_APPROACH_AREA_RATIO or near_edge:
                    self._approach_phase = 'BLIND'
                    self._blind_deadline = now + Duration(seconds=self.BLIND_APPROACH_DURATION)
                    state = 'APPROACHING (entrando a tramo ciego)'
                    twist_msg.linear.x = self.BLIND_APPROACH_LINEAR
                    twist_msg.angular.z = 0.0

                elif area_ratio < self.TARGET_AREA_RATIO:
                    state = 'APPROACHING'
                    area_error = self.TARGET_AREA_RATIO - area_ratio
                    twist_msg.linear.x = float(np.clip(
                        area_error * self.Kp_approach_linear, 0.0, self.MAX_LINEAR))
                    twist_msg.angular.z = float(np.clip(
                        error_x * self.Kp_approach_correction, -self.MAX_ANGULAR, self.MAX_ANGULAR))
                else:
                    # Llegó al área objetivo por visión sin necesitar el
                    # tramo ciego (posible si TARGET_AREA_RATIO se alcanza
                    # con el bbox aún lejos de los bordes).
                    self._approach_phase = 'DONE'
                    state = 'ARRIVED'
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.0

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
                    # medio pasillo: nos comprometemos al MISMO tramo ciego
                    # que ya usamos cuando la visión anticipa que está por
                    # perderse (ver BLIND_APPROACH_* / rama `if area_ratio
                    # >= ...` arriba) -- aquí el disparo es la pérdida total
                    # en lugar del área/borde, pero el destino es idéntico:
                    # avanzar derecho, a ciegas, el tiempo ya calibrado.
                    self._approach_phase = 'BLIND'
                    self._blind_deadline = now + Duration(seconds=self.BLIND_APPROACH_DURATION)
                    state = 'APPROACHING (target perdido, comprometido a tramo ciego)'
                    twist_msg.linear.x = self.BLIND_APPROACH_LINEAR
                    twist_msg.angular.z = 0.0
                    cv2.putText(align_frame, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 165, 255), 2)
                    self.get_logger().warn(f'{state} -> linear={twist_msg.linear.x:.3f}')
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

            self.det_pub.publish(Bool(data=detected))
            self.qr_flag_pub.publish(Bool(data=qr_text is not None))
            if qr_text is not None:
                self.qr_content_pub.publish(String(data=qr_text))
            self.alineacion_pub.publish(Bool(data=(state == 'ARRIVED')))

            self.vel_pub.publish(twist_msg)
            self.publish(annotated_frame, align_frame)
        except Exception as e:
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
