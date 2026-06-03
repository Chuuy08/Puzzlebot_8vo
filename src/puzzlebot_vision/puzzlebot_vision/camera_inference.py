#!/usr/bin/env python3

#Live YOLO inference node — laptop webcam or PuzzleBot camera topic.
# ROS2 node (topic):    ros2 run puzzlebot_vision camera_inference --ros-args -p source:=topic


import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

_MODEL_DEFAULT = os.path.join(
    os.path.dirname(__file__), "..", "models", "yoloN_best.pt"
)

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255),
    (255, 165, 0), (128, 0, 128), (0, 255, 255),
]


def color_for(cls_id: int) -> tuple:
    return COLORS[cls_id % len(COLORS)]


def draw_detections(frame, results, names):
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = color_for(cls_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{names[cls_id]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


# ─── Standalone webcam loop ───────────────────────────────────────────────────

def run_webcam(model_path: str, camera_id: int, conf_thresh: float) -> None:
    from ultralytics import YOLO
    model = YOLO(model_path)
    print(f"[INFO] Modelo cargado: {os.path.basename(model_path)}")
    print(f"[INFO] Clases: {list(model.names.values())}")
    print("[INFO] Presiona 'q' para salir.")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara {camera_id}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_counter = 0
    fps_display = 0.0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        results = model(frame, conf=conf_thresh, verbose=False)[0]
        inference_ms = (time.perf_counter() - t0) * 1000

        annotated = draw_detections(frame.copy(), results, model.names)

        fps_counter += 1
        elapsed = time.time() - t_start
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            t_start = time.time()

        cv2.putText(annotated, f"FPS: {fps_display:.1f}  |  {inference_ms:.0f}ms",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("PuzzleBot Vision — laptop", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─── ROS2 node (camera topic) ─────────────────────────────────────────────────

def run_ros2_node(model_path: str, conf_thresh: float) -> None:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    from ultralytics import YOLO

    class VisionNode(Node):
        def __init__(self):
            super().__init__("puzzlebot_vision_node")
            self.declare_parameter("conf_thresh", conf_thresh)
            self.declare_parameter("model_path", model_path)
            self.declare_parameter("camera_topic", "/camera/image_raw")

            _model_path = self.get_parameter("model_path").get_parameter_value().string_value
            _conf = self.get_parameter("conf_thresh").get_parameter_value().double_value
            _topic = self.get_parameter("camera_topic").get_parameter_value().string_value

            self.model = YOLO(_model_path)
            self.bridge = CvBridge()
            self._conf = _conf

            self.sub = self.create_subscription(Image, _topic, self._cb, 10)
            self.pub = self.create_publisher(Image, "/puzzlebot_vision/detections", 10)
            self.get_logger().info(f"Escuchando en {_topic}")

        def _cb(self, msg: Image):
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            results = self.model(frame, conf=self._conf, verbose=False)[0]
            annotated = draw_detections(frame.copy(), results, self.model.names)
            out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            out_msg.header = msg.header
            self.pub.publish(out_msg)

    rclpy.init()
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live YOLO inference — webcam o ROS2 topic")
    parser.add_argument("--model", default=_MODEL_DEFAULT)
    parser.add_argument("--source", default="0",
                        help="'0' para webcam laptop, 'topic' para ROS2, o índice numérico")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Umbral de confianza (default: 0.4)")
    args = parser.parse_args()

    model_path = os.path.realpath(args.model)
    if not os.path.isfile(model_path):
        print(f"[ERROR] Modelo no encontrado: {model_path}")
        sys.exit(1)

    if args.source == "topic":
        run_ros2_node(model_path, args.conf)
    else:
        try:
            cam_id = int(args.source)
        except ValueError:
            cam_id = 0
        run_webcam(model_path, cam_id, args.conf)


if __name__ == "__main__":
    main()
