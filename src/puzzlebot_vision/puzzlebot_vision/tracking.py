#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
import cv2
import os
import numpy as np
from ament_index_python import get_package_share_directory
from pyzbar.pyzbar import decode as zbar_decode

class Utils(Node):
    def __init__(self):
        super().__init__('yolo_metrics_node')
        try: 
            from ultralytics import YOLO
        except ImportError:
            self.get_logger().error('Ultralytics YOLO library not found"')
            raise
        model_path = os.path.join(get_package_share_directory('puzzlebot_vision'),'models','fine_tunned.pt')
        self.model = YOLO(model_path)
        self.subscription = self.create_subscription(
            CompressedImage,
            '/video_source/compressed', 
            self.image_callback,
            10
        )
        
        self.class_ID = 3
        self.qr_class_ID = 4
        self.model_name = "yolov26n"
        self.lower_blue = np.array([90, 50, 50])
        self.upper_blue = np.array([130, 255, 255])
        self.MAX_SPEED = 0.05

        self.publisher = self.create_publisher(CompressedImage, '/annotated_yolo/compressed', 10)
        self.al_pub = self.create_publisher(CompressedImage, '/align/compressed', 10)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('Alignment testing')

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

    def image_callback(self, msg):
        try:
            twist_msg = Twist()

            # Decodificar CompressedImage
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                return

            h, w, _ = cv_image.shape
            h_center, w_center = h//2, w//2
            
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
                            x = self.decode(roi)
                            if x is not None:
                                self.get_logger().info(f'QR: {x}')
                        break
                if best_box is not None:
                    break
            
            if best_box is not None:
                center_x = (best_box[0] + best_box[2]) // 2
                center_y = (best_box[1] + best_box[3]) // 2
                cv2.circle(alignment_frame, (center_x, center_y), 8, (0, 0, 255), -1)

                error_x = float(w_center - center_x) / (w / 2.0)
                error_y = float(h_center - center_y) / (h / 2.0)
                Kp = 0.1
                Kp_ = 0.1
                ang_z = error_x * Kp
                linear = error_y * Kp_
                twist_msg.linear.x = np.clip(linear, -self.MAX_SPEED, self.MAX_SPEED)
                twist_msg.angular.z = np.clip(ang_z, -self.MAX_SPEED, self.MAX_SPEED)
                
                self.publish(annotated_frame, alignment_frame)
            else:
                self.get_logger().info(f'Target class {self.class_ID} with QR above not found.')
                self.publish(annotated_frame, alignment_frame)
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0

            self.vel_pub.publish(twist_msg)

        except Exception as e:
            self.get_logger().error(f'Inference failed: {e}')
        
    def publish(self, annotated, alignment):
        # Publicar como CompressedImage
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
        node = Utils()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    main()