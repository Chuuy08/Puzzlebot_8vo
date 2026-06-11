import os
import math
import numpy as np
import cv2

# ROS 2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage  
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String, Float32
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import qos_profile_sensor_data

class PoseNode(Node):
    def __init__(self):
        super().__init__('yolo_metrics_node')
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
        self.EnR_sub = self.create_subscription(
            Float32, 
            '/VelocityEncR',
            self.EncRCb,
            qos_profile_sensor_data
        )
        self.EnL_sub = self.create_subscription(
            Float32, 
            '/VelocityEncL',
            self.EncLCb,
            qos_profile_sensor_data
        )
        
        self.class_ID = 3
        self.qr_class_ID = 5
        self.model_name = "yolov26n"
        self.lower_blue = np.array([90, 50, 50])
        self.upper_blue = np.array([130, 255, 255])
        self.MAX_SPEED = 0.05
        
        self.wr_val_ = 0.0
        self.wl_val_ = 0.0

        # ---- Variables para odometría / Localisation ----
        self.first_ = True
        self.last_time_ = self.get_clock().now()
        self.sample_time_ = 0.02 # 20 ms
        self.r_ = 0.05   # Radio de la rueda (Ajusta según tu Puzzlebot)
        self.l_ = 0.19   # Distancia entre ruedas (Ajusta según tu Puzzlebot)
        self.X_ = 0.0
        self.Y_ = 0.0
        self.Th_ = 0.0
        self.V_ = 0.0
        self.Omega_ = 0.0

        # ---- Variables del Control de Estados ----
        # "ALIGN", "FORWARD", "FINISHED"
        self.alignment_threshold = 0.1 # Umbral de tolerancia de error (ajustable)
        self.target_distance = 0.48      # 40 centímetros en metros

        # Reutilizado para las 4 áreas de pickup; sin reset, state='FINISHED'
        # quedaría fantasma en los siguientes intentos. Mismo contrato de
        # reset que align_and_approach (/align_and_approach/reset).
        self._reset_state()

        self.reset_subscription = self.create_subscription(
            Bool,
            '/align_and_approach/reset',
            self._reset_callback,
            10
        )

        self.publisher = self.create_publisher(CompressedImage, '/annotated_yolo/compressed', 10)
        self.al_pub = self.create_publisher(CompressedImage, '/align/compressed', 10)
        
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('Alignment and 40cm Forward test initiated.')

        self.det_pub = self.create_publisher(Bool, '/pallet_detected', 10)
        self.qr_flag_pub = self.create_publisher(Bool, '/pallet_has_qr', 10)
        self.qr_content_pub = self.create_publisher(String, '/pallet_qr_content', 10)
        self.alineacion_pub = self.create_publisher(Bool, '/alineation/booleano', 10)

    def EncLCb(self, msg):
        self.wl_val_ = msg.data

    def EncRCb(self, msg):
        self.wr_val_ = msg.data

    def _reset_state(self):
        """Reinicia la FSM de alineación/avance. start_x/start_y se anclan a
        la posición ACTUAL para que distance_travelled se mida desde aquí."""
        self.state = "ALIGN"
        self.start_x = self.X_
        self.start_y = self.Y_

    def _reset_callback(self, msg):
        self.get_logger().info(
            'Reset recibido -> reiniciando estado para nuevo intento de aproximación')
        self._reset_state()

    def decode(self, roi):
        try:
            from pyzbar.pyzbar import decode as zbar_decode
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

    def image_callback(self, msg):
        try:
            twist_msg = Twist()
            # Constantly update coordinates from encoder tracking
            self.localisation()
            
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is None:
                self.get_logger().error("No se pudo decodificar la imagen comprimida.")
                return

            h, w, _ = cv_image.shape
            w_center = w // 2
            
            results = self.model(cv_image, device='cpu', verbose=False)
            annotated_frame = results[0].plot()
            alignment_frame = annotated_frame.copy()
            cv2.line(alignment_frame, (w_center, 0), (w_center, h), (0, 255, 0), 2)
            
            boxes_3 = []
            boxes_4 = []
            
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                coords = box.xyxy[0].cpu().numpy().astype(int)
                if cls_id == self.class_ID:
                    boxes_3.append(coords)
                elif cls_id == self.qr_class_ID:
                    boxes_4.append(coords)
            
            best_box = None
            qr_content_string = ""
            pallet_has_qr = False
            
            for b3 in boxes_3:
                xmin3, ymin3, xmax3, ymax3 = b3
                center_x3 = (xmin3 + xmax3) // 2
                
                for b4 in boxes_4:
                    xmin4, ymin4, xmax4, ymax4 = b4
                    center_x4 = (xmin4 + xmax4) // 2
                    
                    if ymax4 <= (ymin3 + 20) and (xmin3 <= center_x4 <= xmax3 or xmin4 <= center_x3 <= xmax4):
                        best_box = b3
                        
                        ymin4_safe = max(0, ymin4)
                        ymax4_safe = min(h, ymax4)
                        xmin4_safe = max(0, xmin4)
                        xmax4_safe = min(w, xmax4)
                        
                        if ymax4_safe > ymin4_safe and xmax4_safe > xmin4_safe:
                            roi = cv_image[ymin4_safe:ymax4_safe, xmin4_safe:xmax4_safe]
                            decoded_data = self.decode(roi)
                            if decoded_data is not None:
                                self.get_logger().info(f'QR Decoded: {decoded_data}')
                                qr_content_string = decoded_data
                                pallet_has_qr = True
                        break
                if best_box is not None:
                    break
            
            det_msg = Bool()
            qr_flag_msg = Bool()
            qr_str_msg = String()
            alineacion_msg = Bool()

            if self.state == "ALIGN":
                if best_box is not None:
                    det_msg.data = True
                    qr_flag_msg.data = pallet_has_qr
                    qr_str_msg.data = qr_content_string
                    alineacion_msg.data = False

                    center_x = (best_box[0] + best_box[2]) // 2
                    center_y = (best_box[1] + best_box[3]) // 2
                    cv2.circle(alignment_frame, (center_x, center_y), 8, (0, 0, 255), -1)

                    error_x = float(w_center - center_x) / (w / 2.0)
                    Kp = 0.1
                    ang_z = error_x * Kp
                    
                    # Rotates in place until it is centered
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = np.clip(ang_z, -self.MAX_SPEED, self.MAX_SPEED)
                    
                    if abs(error_x) < self.alignment_threshold:
                        self.get_logger().info('¡Robot Inicialmente Alineado! Iniciando avance...')
                        self.state = "FORWARD"
                        self.start_x = self.X_
                        self.start_y = self.Y_
                    
                    self.publish(annotated_frame, alignment_frame)
                else:
                    self.get_logger().info('Buscando objetivo para iniciar alineación...', throttle_duration_sec=1.0)
                    det_msg.data = False
                    qr_flag_msg.data = False
                    qr_str_msg.data = ""
                    alineacion_msg.data = False
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.0
                    self.publish_empty_frame(annotated_frame)

            elif self.state == "FORWARD":
                # NO publicar True aquí: solo se completó el centrado angular,
                # falta el avance hasta target_distance. Este topic dispara en
                # paralelo el montacargas (fpga_controller) y mission_manager
                # (WAITING_LOAD) -- solo FINISHED reporta alineación completa.
                alineacion_msg.data = False
                det_msg.data = True

                # Compute distance travelled from the initial baseline coordinate
                distance_travelled = math.sqrt((self.X_ - self.start_x)**2 + (self.Y_ - self.start_y)**2)
                
                if distance_travelled >= self.target_distance: # Reached 40 cm
                    self.get_logger().info('¡Meta total de 40cm alcanzada! Frenando robot.')
                    self.state = "FINISHED"
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.0

                elif distance_travelled < 0.10: # FIRST 10 CM: Drive forward + Visual fine-tuning
                    self.get_logger().info(f'Fase 1 (0-10cm): Avanzando y Alineando. Recorrido: {distance_travelled:.3f} m', throttle_duration_sec=0.5)
                    
                    twist_msg.linear.x = 0.03 # Move forward slowly 
                    
                    if best_box is not None:
                        center_x = (best_box[0] + best_box[2]) // 2
                        error_x = float(w_center - center_x) / (w / 2.0)
                        Kp = 0.08 # Slightly gentler controller gain to prevent wild swaying while moving
                        ang_z = error_x * Kp
                        twist_msg.angular.z = np.clip(ang_z, -self.MAX_SPEED, self.MAX_SPEED)
                    else:
                        twist_msg.angular.z = 0.0 # Maintain course if target is temporarily lost

                else: # NEXT 30 CM (From 10cm to 40cm): Purely blind linear displacement
                    self.get_logger().info(f'Fase 2 (10-40cm): Avance ciego rectilineo. Recorrido: {distance_travelled:.3f} m', throttle_duration_sec=0.5)
                    twist_msg.linear.x = 0.04 # Can move slightly quicker now
                    twist_msg.angular.z = 0.0

                self.publish(annotated_frame, alignment_frame)

            elif self.state == "FINISHED":
                alineacion_msg.data = True
                det_msg.data = False
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0
                self.publish(annotated_frame, alignment_frame)

            self.det_pub.publish(det_msg)
            self.qr_flag_pub.publish(qr_flag_msg)
            self.qr_content_pub.publish(qr_str_msg)
            self.alineacion_pub.publish(alineacion_msg)
            self.vel_pub.publish(twist_msg)

        except Exception as e:
            self.get_logger().error(f'Inference failed: {e}')
        
    def publish(self, annotated, alignment):
        msg_annotated = CompressedImage()
        msg_annotated.header.stamp = self.get_clock().now().to_msg()
        msg_annotated.format = "jpeg"
        msg_annotated.data = cv2.imencode('.jpg', annotated)[1].tobytes()

        msg_alignment = CompressedImage()
        msg_alignment.header.stamp = self.get_clock().now().to_msg()
        msg_alignment.format = "jpeg"
        msg_alignment.data = cv2.imencode('.jpg', alignment)[1].tobytes()

        self.publisher.publish(msg_annotated)
        self.al_pub.publish(msg_alignment)

    def publish_empty_frame(self, annotated_frame):
        out_msg = CompressedImage()
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.format = "jpeg"
        out_msg.data = cv2.imencode('.jpg', annotated_frame)[1].tobytes()
        self.publisher.publish(out_msg)
        
    def localisation(self):
        current_time = self.get_clock().now()
        
        if self.first_:
            self.last_time_ = current_time
            self.first_ = False
            return

        dt = (current_time - self.last_time_).nanoseconds / 1e9

        if dt > self.sample_time_:
            v_r = self.r_ * self.wr_val_
            v_l = self.r_ * self.wl_val_
            velocity_threshold = 1e-3

            if abs(v_r) < velocity_threshold and abs(v_l) < velocity_threshold:
                self.V_ = 0.0
                self.Omega_ = 0.0
                self.last_time_ = current_time
                return

            self.V_ = (v_r + v_l) / 2.0
            self.Omega_ = (v_r - v_l) / self.l_

            delta_theta = self.Omega_ * dt
            self.Th_ += delta_theta
            self.Th_ = self.wrap_to_pi(self.Th_)

            self.X_ += self.V_ * math.cos(self.Th_) * dt
            self.Y_ += self.V_ * math.sin(self.Th_) * dt
            
            self.last_time_ = current_time

    def wrap_to_pi(self, theta):
        result = math.fmod((theta + math.pi), (2.0 * math.pi))
        if result < 0:
            result += (2.0 * math.pi)
        return result - math.pi

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PoseNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()