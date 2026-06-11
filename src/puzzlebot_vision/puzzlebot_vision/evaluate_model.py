#!/usr/bin/env python3
# YOLO model evaluator: inspection, precision, speed, and Jetson Nano viability report.


import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch

# ─── Paths ────────────────────────────────────────────────────────────────────

_PKG_ROOT = os.path.join(os.path.dirname(__file__), "..")
_MODEL_DEFAULT = os.path.join(
    _PKG_ROOT, "models", "yoloN_best.pt"
)
_VIDEO_DEFAULT = os.path.join(
    _PKG_ROOT, "media", "video", "model_test.mp4"
)
_RESULTS_DIR = os.path.join(_PKG_ROOT, "resultados")

# ─── Helpers ──────────────────────────────────────────────────────────────────

SEP = "─" * 70
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255),
    (255, 165, 0), (128, 0, 128), (0, 255, 255),
]


def banner(title: str) -> None:
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print('═' * 70)


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def ram_usage_mb() -> float:
    """Current process RSS in MB."""
    try:
        import resource as _r
        return _r.getrusage(_r.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return 0.0


def vram_usage_mb() -> float:
    """Current CUDA memory allocated in MB (0 if no GPU)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 2
    return 0.0


def color_for_class(class_id: int) -> tuple:
    return COLORS[class_id % len(COLORS)]


# ─── 1. Model Inspection ──────────────────────────────────────────────────────

def inspect_model(model_path: str) -> dict:
    banner("1. INSPECCIÓN DEL MODELO")
    from ultralytics import YOLO

    model = YOLO(model_path)
    info = {}

    # Architecture name from model string representation
    model_name = type(model.model).__name__
    yaml_name = getattr(model.model, "yaml", {}).get("yaml_file", "desconocido")
    num_layers = len(list(model.model.modules()))
    num_params = sum(p.numel() for p in model.model.parameters())
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

    # Detect YOLO version from model name
    version_tag = "desconocida"
    for tag in ["YOLOv8", "YOLOv9", "YOLOv10", "YOLOv11", "YOLO26", "YOLO"]:
        if tag.lower() in model_name.lower() or tag.lower() in str(yaml_name).lower():
            version_tag = tag
            break

    # Size category (nano/small/medium/large/xlarge) from param count
    size_map = [
        (3_000_000, "nano (n)"),
        (10_000_000, "small (s)"),
        (25_000_000, "medium (m)"),
        (50_000_000, "large (l)"),
        (float("inf"), "xlarge (x)"),
    ]
    size_label = next(lbl for thr, lbl in size_map if num_params <= thr)

    model_file_mb = os.path.getsize(model_path) / 1024 ** 2

    print(f"  Archivo           : {os.path.basename(model_path)}")
    print(f"  Tamaño en disco   : {model_file_mb:.2f} MB")
    print(f"  Arquitectura base : {model_name}")
    print(f"  Versión detectada : {version_tag}")
    print(f"  Categoría tamaño  : {size_label}")
    print(f"  Capas totales     : {num_layers}")
    print(f"  Parámetros totales: {num_params:,}")
    print(f"  Parámetros entren.: {trainable:,}")
    print(f"  Task              : {model.task}")

    print(f"\n  Clases detectadas ({len(model.names)}):")
    for idx, name in model.names.items():
        print(f"    [{idx}] {name}")

    info.update({
        "model": model,
        "model_path": model_path,
        "model_name": model_name,
        "num_params": num_params,
        "model_file_mb": model_file_mb,
        "classes": model.names,
        "version_tag": version_tag,
        "size_label": size_label,
    })
    return info


# ─── 2. Precision Evaluation ──────────────────────────────────────────────────

def evaluate_precision(model, video_path: str, results_dir: str) -> dict:
    banner("2. EVALUACIÓN DE PRECISIÓN")
    os.makedirs(results_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] No se pudo abrir el video: {video_path}")
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Video             : {os.path.basename(video_path)}")
    print(f"  Resolución        : {width}x{height}")
    print(f"  FPS del video     : {fps_video:.1f}")
    print(f"  Frames totales    : {total_frames}")
    print()

    # ── Output video writer ──────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(results_dir, "detecciones.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps_video, (width, height))

    class_stats: dict[str, list] = {}
    total_detections = 0
    frame_count = 0
    sample_saved = 0
    sample_every = max(1, total_frames // 20)  # save ~20 sample frames

    print("  Procesando video", end="", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model(frame, verbose=False)[0]
        annotated = frame.copy()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = color_for_class(cls_id)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            class_stats.setdefault(cls_name, []).append(conf)
            total_detections += 1

        writer.write(annotated)

        # Save sample frame as image
        if frame_count % sample_every == 0:
            img_path = os.path.join(results_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(img_path, annotated)
            sample_saved += 1

        if frame_count % 50 == 0:
            print(".", end="", flush=True)

    cap.release()
    writer.release()
    print(f" ✓ ({frame_count} frames)")

    print(f"\n  Frames procesados : {frame_count}")
    print(f"  Detecciones total : {total_detections}")
    print(f"  Video guardado en : {out_path}")
    print(f"  Frames muestra    : {sample_saved} imgs en {results_dir}/")

    print("\n  Estadísticas de confianza por clase:")
    print(f"  {'Clase':<12} {'Detecciones':>12} {'Conf media':>12} {'Conf max':>10} {'Conf min':>10}")
    print("  " + "-" * 60)
    for cls_name, confs in sorted(class_stats.items()):
        arr = np.array(confs)
        print(f"  {cls_name:<12} {len(arr):>12} {arr.mean():>11.3f}  {arr.max():>9.3f}  {arr.min():>9.3f}")

    return {
        "frames_processed": frame_count,
        "total_detections": total_detections,
        "class_stats": class_stats,
        "output_video": out_path,
    }


# ─── 3. Speed Evaluation ──────────────────────────────────────────────────────

def _measure_inference_times(model, frames: list, device: str) -> list[float]:
    """Return per-frame inference times in ms."""
    times = []
    for frame in frames:
        t0 = time.perf_counter()
        _ = model(frame, verbose=False, device=device)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def evaluate_speed(model, video_path: str, n_frames: int = 50) -> dict:
    banner("3. EVALUACIÓN DE VELOCIDAD (laptop → Jetson Nano)")

    # Collect sample frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < n_frames:
        print(f"  [AVISO] Video tiene solo {len(frames)} frames; se usarán todos.")
    actual = len(frames)

    results = {}

    # ── CPU ──────────────────────────────────────────────────────────────────
    section("3a. CPU Inference")
    ram_before = ram_usage_mb()
    # Warmup
    for f in frames[:3]:
        model(f, verbose=False, device="cpu")

    times_cpu = _measure_inference_times(model, frames, device="cpu")
    ram_after = ram_usage_mb()

    p50 = np.percentile(times_cpu, 50)
    p95 = np.percentile(times_cpu, 95)
    mean_cpu = np.mean(times_cpu)

    print(f"  Frames analizados : {actual}")
    print(f"  Tiempo medio      : {mean_cpu:.1f} ms/frame")
    print(f"  Mediana (p50)     : {p50:.1f} ms/frame")
    print(f"  Percentil 95      : {p95:.1f} ms/frame")
    print(f"  FPS estimado      : {1000/mean_cpu:.1f}")
    print(f"  RAM usada (delta) : {ram_after - ram_before:.1f} MB  (total RSS: {ram_after:.1f} MB)")
    results["cpu"] = {"mean_ms": mean_cpu, "fps": 1000/mean_cpu, "p95_ms": p95}

    # ── CUDA ─────────────────────────────────────────────────────────────────
    section("3b. CUDA / GPU Inference")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  GPU detectada     : {gpu_name}")
        torch.cuda.reset_peak_memory_stats()
        for f in frames[:3]:
            model(f, verbose=False, device="cuda")

        times_cuda = _measure_inference_times(model, frames, device="cuda")
        vram = torch.cuda.max_memory_allocated() / 1024 ** 2
        mean_cuda = np.mean(times_cuda)
        p95_cuda = np.percentile(times_cuda, 95)

        print(f"  Tiempo medio      : {mean_cuda:.1f} ms/frame")
        print(f"  Percentil 95      : {p95_cuda:.1f} ms/frame")
        print(f"  FPS estimado      : {1000/mean_cuda:.1f}")
        print(f"  VRAM pico usada   : {vram:.1f} MB")
        results["cuda"] = {"mean_ms": mean_cuda, "fps": 1000/mean_cuda, "vram_mb": vram}
    else:
        print("  [INFO] CUDA no disponible en este equipo.")
        print("  → En Jetson Nano (CUDA 10.2) se esperan mejoras significativas.")
        results["cuda"] = None

    # ── ONNX ─────────────────────────────────────────────────────────────────
    section("3c. ONNX Runtime Inference")
    try:
        import onnxruntime as ort
        onnx_available = True
    except ImportError:
        onnx_available = False
        print("  [INFO] onnxruntime no instalado.")
        print("  → Instala con: pip install onnxruntime")
        results["onnx"] = None

    if onnx_available:
        import shutil
        models_dir = os.path.join(_PKG_ROOT, "models")
        os.makedirs(models_dir, exist_ok=True)
        exported = str(model.export(format="onnx", imgsz=640, simplify=True, verbose=False))
        onnx_dest = os.path.join(models_dir, os.path.basename(exported))
        if os.path.realpath(exported) != os.path.realpath(onnx_dest):
            shutil.move(exported, onnx_dest)
        onnx_path = onnx_dest
        print(f"  Modelo ONNX       : {onnx_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(onnx_path, sess_options,
                                    providers=["CPUExecutionProvider"])

        input_name = sess.get_inputs()[0].name
        input_shape = [1, 3, 640, 640]

        def preprocess(frame):
            img = cv2.resize(frame, (640, 640))
            img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            return img[np.newaxis, ...]

        # Warmup
        for f in frames[:3]:
            blob = preprocess(f)
            sess.run(None, {input_name: blob})

        times_onnx = []
        for f in frames:
            blob = preprocess(f)
            t0 = time.perf_counter()
            sess.run(None, {input_name: blob})
            times_onnx.append((time.perf_counter() - t0) * 1000)

        mean_onnx = np.mean(times_onnx)
        p95_onnx = np.percentile(times_onnx, 95)
        speedup = mean_cpu / mean_onnx if mean_cpu > 0 else 1.0

        print(f"  Tiempo medio      : {mean_onnx:.1f} ms/frame")
        print(f"  Percentil 95      : {p95_onnx:.1f} ms/frame")
        print(f"  FPS estimado      : {1000/mean_onnx:.1f}")
        print(f"  Speedup vs CPU    : {speedup:.2f}x")
        results["onnx"] = {"mean_ms": mean_onnx, "fps": 1000/mean_onnx, "speedup": speedup}

    return results


# ─── 4. Final Report ──────────────────────────────────────────────────────────

def final_report(info: dict, precision: dict, speed: dict) -> None:
    banner("4. REPORTE FINAL — VIABILIDAD PARA JETSON NANO")

    min_rt_fps = 15.0
    cpu_fps = speed.get("cpu", {}).get("fps", 0)
    cuda_fps = speed.get("cuda", {}).get("fps", 0) if speed.get("cuda") else 0
    onnx_fps = speed.get("onnx", {}).get("fps", 0) if speed.get("onnx") else 0
    best_laptop_fps = max(cpu_fps, cuda_fps, onnx_fps)

    num_params = info.get("num_params", 0)
    model_mb = info.get("model_file_mb", 0)
    size_label = info.get("size_label", "?")
    classes = info.get("classes", {})

    # ── Velocidad ─────────────────────────────────────────────────────────────
    section("4a. Velocidad")
    print(f"  CPU (laptop)      : {cpu_fps:.1f} FPS")
    if cuda_fps:
        print(f"  CUDA (laptop)     : {cuda_fps:.1f} FPS")
    if onnx_fps:
        print(f"  ONNX (laptop)     : {onnx_fps:.1f} FPS")

    # Estimaciones conservadoras Jetson Nano (CPU≈laptop CPU/2, TRT INT8≈CPU×1.8)
    jetson_cpu_est = cpu_fps * 0.5
    jetson_trt_est = cpu_fps * 1.8

    print(f"\n  Estimaciones Jetson Nano:")
    print(f"    CPU puro          : ~{jetson_cpu_est:.1f} FPS")
    print(f"    CUDA (sin TRT)    : ~{jetson_cpu_est * 2:.1f} FPS")
    print(f"    TensorRT (INT8)   : ~{jetson_trt_est:.1f} FPS  ← recomendado")

    rt_viable = jetson_trt_est >= min_rt_fps
    symbol = "✓" if rt_viable else "✗"
    print(f"\n  [{symbol}] Tiempo real (>{min_rt_fps} FPS) en Jetson Nano con TRT: "
          f"{'VIABLE' if rt_viable else 'MARGINAL — considera optimizar'}")

    # ── Precisión ─────────────────────────────────────────────────────────────
    section("4b. Precisión detectada")
    class_stats = precision.get("class_stats", {})
    if class_stats:
        for cls_name, confs in sorted(class_stats.items()):
            arr = np.array(confs)
            quality = "alta" if arr.mean() > 0.7 else ("media" if arr.mean() > 0.5 else "baja")
            print(f"  {cls_name:<12}: conf media {arr.mean():.3f}  → confianza {quality}")
    else:
        print("  Sin detecciones en el video de prueba.")

    # ── TensorRT ──────────────────────────────────────────────────────────────
    section("4c. Recomendación TensorRT para Jetson Nano")
    print("  El modelo usa la arquitectura YOLO26n (nano).")
    print("  Con TensorRT FP16/INT8 en Jetson Nano se logra ~3-4× speedup.")
    print()
    print("  Pasos para exportar a TensorRT:")
    print("    # En la Jetson Nano (JetPack ≥4.6):")
    print("    from ultralytics import YOLO")
    print("    model = YOLO('yoloN_best.pt')")
    print("    model.export(format='engine', half=True, device=0)")
    print()
    print("  Alternativa rápida con ONNX:")
    print("    model.export(format='onnx', simplify=True)")
    print("    # Luego convertir con trtexec:")
    print("    # trtexec --onnx=yoloN_best.onnx --saveEngine=yoloN_best.engine --fp16")

    # ── Tamaño del modelo ─────────────────────────────────────────────────────
    section("4d. ¿Conviene afinar o cambiar el modelo?")
    print(f"  Parámetros  : {num_params:,}  ({size_label})")
    print(f"  Tamaño .pt  : {model_mb:.2f} MB")
    print(f"  Clases      : {len(classes)} ({', '.join(classes.values())})")
    print()

    if num_params <= 3_000_000:
        print("  El modelo nano es la opción más ligera disponible.")
        print("  → Mantenerlo si la precisión es suficiente para el caso de uso.")
        print("  → Considerar YOLOv8n o MobileNet-SSD si se necesita aún menos latencia.")
    elif num_params <= 10_000_000:
        print("  El modelo small es equilibrado pero puede ser pesado para Jetson Nano.")
        print("  → Probar la variante nano (n) si la precisión no se degrada.")
    else:
        print("  El modelo es demasiado grande para tiempo real en Jetson Nano.")
        print("  → Reentrenar con YOLOv8n o afinar con dataset reducido.")

    print()
    print("  RESUMEN EJECUTIVO:")
    print(f"    • Laptop CPU     : {cpu_fps:.1f} FPS")
    if rt_viable:
        print(f"    • Jetson + TRT   : ~{jetson_trt_est:.1f} FPS  → VIABLE para tiempo real")
    else:
        print(f"    • Jetson + TRT   : ~{jetson_trt_est:.1f} FPS  → puede ser insuficiente")
    print("    • Exportar a TensorRT FP16 es ALTAMENTE recomendado para Jetson Nano")
    print("    • Sin TRT la Jetson Nano probablemente no alcance los 15 FPS mínimos")
    print()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluación completa de modelo YOLO para PuzzleBot / Jetson Nano"
    )
    parser.add_argument("--model", default=_MODEL_DEFAULT,
                        help="Ruta al archivo .pt del modelo")
    parser.add_argument("--video", default=_VIDEO_DEFAULT,
                        help="Ruta al video de prueba")
    parser.add_argument("--results-dir", default=_RESULTS_DIR,
                        help="Carpeta para guardar resultados")
    parser.add_argument("--speed-frames", type=int, default=50,
                        help="Número de frames para benchmark de velocidad")
    parser.add_argument("--skip-precision", action="store_true",
                        help="Omitir la evaluación de precisión (más rápido)")
    parser.add_argument("--skip-speed", action="store_true",
                        help="Omitir la evaluación de velocidad")
    args = parser.parse_args()

    # Resolve paths relative to this file if not absolute
    model_path = os.path.realpath(args.model)
    video_path = os.path.realpath(args.video)
    results_dir = os.path.realpath(args.results_dir)

    if not os.path.isfile(model_path):
        print(f"[ERROR] Modelo no encontrado: {model_path}")
        sys.exit(1)
    if not os.path.isfile(video_path):
        print(f"[ERROR] Video no encontrado: {video_path}")
        sys.exit(1)

    os.makedirs(results_dir, exist_ok=True)

    # ── Run stages ────────────────────────────────────────────────────────────
    info = inspect_model(model_path)
    model = info["model"]

    precision = {}
    if not args.skip_precision:
        precision = evaluate_precision(model, video_path, results_dir)

    speed = {}
    if not args.skip_speed:
        speed = evaluate_speed(model, video_path, n_frames=args.speed_frames)

    final_report(info, precision, speed)


if __name__ == "__main__":
    main()
