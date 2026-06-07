#!/usr/bin/env python3
import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
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

        model_path = os.path.join(get_package_share_directory('puzzlebot_vision'), 'models', 'fine_tunned.pt')
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

        self.class_ID = 3
        self.qr_class_ID = 4

        # Phase 1 (ALIGNING): rotate in place until the target is centered horizontally.
        self.ALIGN_ERROR_THRESHOLD = 0.08
        self.Kp_align = 0.25

        # Phase 2 (APPROACHING): drive forward using bbox area (image coverage) as a
        # distance proxy, with a small angular correction to stay centered.
        self.TARGET_AREA_RATIO = 0.35
        self.Kp_approach_linear = 0.6
        self.Kp_approach_correction = 0.1

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
        boxes_3 = []
        boxes_4 = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            coords = box.xyxy[0].cpu().numpy().astype(int)
            if cls_id == self.class_ID:
                boxes_3.append(coords)
            elif cls_id == self.qr_class_ID:
                boxes_4.append(coords)

        for b3 in boxes_3:
            xmin3, ymin3, xmax3, ymax3 = b3
            center_x3 = (xmin3 + xmax3) // 2
            for b4 in boxes_4:
                xmin4, ymin4, xmax4, ymax4 = b4
                center_x4 = (xmin4 + xmax4) // 2
                if ymax4 <= (ymin3 + 20) and (xmin3 <= center_x4 <= xmax3 or xmin4 <= center_x3 <= xmax4):
                    qr_roi = (max(0, xmin4), max(0, ymin4), min(w, xmax4), min(h, ymax4))
                    return b3, qr_roi
        return None, None

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
            best_box, qr_roi = self.find_target(results, h, w)

            detected = best_box is not None
            qr_text = None
            state = None

            if best_box is not None:
                if qr_roi is not None:
                    xmin4, ymin4, xmax4, ymax4 = qr_roi
                    if ymax4 > ymin4 and xmax4 > xmin4:
                        qr_text = self.decode(cv_image[ymin4:ymax4, xmin4:xmax4])
                        if qr_text is not None:
                            self.get_logger().info(f'QR: {qr_text}')

                xmin3, ymin3, xmax3, ymax3 = best_box
                center_x = (xmin3 + xmax3) // 2
                center_y = (ymin3 + ymax3) // 2
                cv2.circle(align_frame, (center_x, center_y), 8, (0, 0, 255), -1)

                error_x = float(w_center - center_x) / (w / 2.0)
                area_ratio = float((xmax3 - xmin3) * (ymax3 - ymin3)) / float(w * h)

                if abs(error_x) > self.ALIGN_ERROR_THRESHOLD:
                    state = 'ALIGNING'
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = float(np.clip(
                        error_x * self.Kp_align, -self.MAX_ANGULAR, self.MAX_ANGULAR))
                elif area_ratio < self.TARGET_AREA_RATIO:
                    state = 'APPROACHING'
                    area_error = self.TARGET_AREA_RATIO - area_ratio
                    twist_msg.linear.x = float(np.clip(
                        area_error * self.Kp_approach_linear, 0.0, self.MAX_LINEAR))
                    twist_msg.angular.z = float(np.clip(
                        error_x * self.Kp_approach_correction, -self.MAX_ANGULAR, self.MAX_ANGULAR))
                else:
                    state = 'ARRIVED'
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.0

                cv2.putText(align_frame, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 255), 2)
                self.get_logger().info(
                    f'{state} | error_x={error_x:.2f} area_ratio={area_ratio:.2f} '
                    f'-> linear={twist_msg.linear.x:.3f} angular={twist_msg.angular.z:.3f}')
            else:
                self.get_logger().info(f'Target class {self.class_ID} with QR above not found.')
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
