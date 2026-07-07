# PuzzleBot — Sistema Autónomo de Logística con ROS 2

Proyecto final de carrera. Robot móvil diferencial (PuzzleBot, Manchester Robotics) capaz de identificar, recoger y entregar pallets de forma autónoma dentro de un almacén simulado y real, integrando percepción visual, localización probabilística, navegación reactiva y control de actuadores vía FPGA.

---

## Demostración

**Pipeline completo (misión end-to-end):**
<!-- Subir video directamente aquí arrastrando un .mp4 en el editor de GitHub -->

**Evasión de obstáculos (DWA):**
<!-- Subir video directamente aquí arrastrando un .mp4 en el editor de GitHub -->

**Construcción de mapa (SLAM):**
<!-- Subir video directamente aquí arrastrando un .mp4 en el editor de GitHub -->

---

## ¿Qué hace el sistema?

1. Se localiza dentro de un mapa conocido usando un filtro de partículas (MCL) sobre LiDAR y odometría.
2. Planea rutas libres de colisión (RRT bidireccional) y las sigue evitando obstáculos dinámicos (DWA).
3. Detecta pallets y códigos QR en cámara con un modelo YOLO entrenado propio.
4. Se alinea y aproxima al pallet en tres fases: alineación visual → aproximación proporcional al área → tramo ciego por odometría.
5. Acciona un mecanismo de elevación tipo montacargas controlado por una FPGA vía SPI.
6. Repite el ciclo hasta completar la entrega, reportando estado en tiempo real por un dashboard web y respondiendo a comandos de voz.

---

## Arquitectura

```
┌──────────────────────────────────────────────────────────┐
│                     INTERFAZ                              │
│   Dashboard web (gRPC/Protobuf)  |  Comandos de voz (HMM)│
└──────────────▲───────────────────────────▲───────────────┘
               │ telemetría                │ /voice_cmd
┌──────────────┴───────────────────────────┴───────────────┐
│                  MISIÓN (FSM)                             │
│          mission_manager_node  — orquesta todo            │
└───┬──────────────────┬─────────────────────┬─────────────┘
    │ /goal_pose        │ /align_and_approach  │ /deteccion_pallet
┌───▼──────────┐  ┌────▼─────────────┐  ┌────▼──────────────┐
│  NAVEGACIÓN  │  │   PERCEPCIÓN     │  │   FPGA (SPI)       │
│ RRT          │  │ YOLO + QR        │  │ mecanismo elevador │
│ Pure Pursuit │  │ align_and_approach│  │ fpga_controller   │
│ DWA          │  │ tracking         │  └───────────────────┘
│ costmap      │  └──────────────────┘
└───┬──────────┘
    │
┌───▼──────────────────────────────────────────────────────┐
│              LOCALIZACIÓN — MCL                           │
│  filtro de partículas: /odom + /scan + /map → pose + TF  │
└───┬──────────────────────────────────────────────────────┘
    │
┌───▼──────────────────────────────┐   ┌───────────────────┐
│  HARDWARE                        │◄──┤  CONTROL PID      │
│  Encoders · LiDAR · Cámara       │   │  distancia + ang. │
│  Jetson Nano + FPGA              │   └───────────────────┘
└──────────────────────────────────┘
```

---

## Módulos principales

| Módulo | Descripción |
|---|---|
| `puzzlebot_localisation` | SLAM propio (occupancy grid + log-odds), Google Cartographer 2D, MCL (filtro de partículas 1000–1500 partículas), localización activa ante pérdida de convergencia |
| `puzzlebot_navigation` | Costmap (estático + dinámico), planeador global RRT bidireccional, seguidor Pure Pursuit, evasión reactiva DWA |
| `puzzlebot_vision` | Inferencia YOLO, detección y decodificación de QR, FSM de alineación/aproximación en tres fases |
| `puzzlebot_control` | Dos lazos PID en cascada (distancia y orientación) para seguimiento de waypoints |
| `puzzlebot_fpga_controller` | Puente ROS 2 → SPI → FSM en FPGA; secuencias de elevación calibradas por tipo de estación (RACK / RODILLO) |
| `puzzlebot_voice` | Reconocimiento de comandos por HMM (Baum-Welch / Viterbi) con características LPC/LSF |
| `puzzlebot_gui` | Dashboard web en tiempo real (pose, mapa, partículas, ruta, cámara) vía gRPC + Protobuf |
| `puzzlebot_mission` | Máquina de estados finita que orquesta navegación, visión, FPGA y reporte de estado |

### Construcción del mapa — SLAM

El proyecto pasó por dos etapas de mapeo antes de usar el mapa fijo con MCL:

**SLAM personalizado** (`occupancy_grid_node.py`): implementación propia sobre ROS 2 que integra `/odom` + `/scan` actualizando una `OccupancyGrid` con un modelo probabilístico de log-odds. Fue el primer enfoque, desarrollado desde cero.

**Google Cartographer** (`cartographer_ros`, modo 2D): usado como segunda alternativa para obtener mapas de mayor fidelidad geométrica. Configurado solo con LiDAR, sin sensores adicionales. El mapa resultante (`.pgm` + `.yaml`) es el que alimenta el `map_server` en los lanzamientos de MCL.

---

## Tecnologías

| Área | Herramientas |
|---|---|
| Middleware | ROS 2, tf2, nav_msgs, sensor_msgs |
| Lenguajes | Python 3, C (driver SPI), VHDL/Verilog (FSM en FPGA) |
| Visión | OpenCV, YOLO (Ultralytics), pyzbar |
| Mapeo / SLAM | SLAM propio (log-odds), Google Cartographer 2D |
| Localización | Monte Carlo Localization (propio), ICP |
| Navegación | RRT bidireccional, Pure Pursuit, DWA, PID |
| Hardware | NVIDIA Jetson Nano, FPGA (SPI) |
| Voz | HMM, LPC/LSF |
| Simulación | Gazebo (SDF + URDF/Xacro) |
| Telemetría | gRPC, Protocol Buffers, dashboard web |

---

## Ejecución

> Requiere workspace compilado (`colcon build`) y `source install/setup.bash`.

**Simulación:**
```bash
ros2 launch puzzlebot_gazebo gazebo_world_launch.py world:=dt_world.sdf
ros2 launch puzzlebot_localisation mcl_launch.py
ros2 launch puzzlebot_navigation navigation_launch.py
ros2 run puzzlebot_vision align_and_approach
ros2 launch puzzlebot_mission mission_launch.py
```

**Robot real (bring-up completo):**
```bash
ros2 launch puzzlebot_mission full_mission_real_launch.py
```

---

## Resultados

- Localización convergente desde pose inicial desconocida con recuperación activa ante pérdida de señal.
- Navegación punto a punto evitando obstáculos dinámicos no presentes en el mapa estático.
- Pipeline de visión funcional: detección de pallet, lectura de QR y aproximación en tres fases hasta tolerancia de manipulación.
- Integración ROS 2 ↔ SPI ↔ FPGA para actuación mecánica sincronizada con los hitos de misión.
- Misión end-to-end ejecutada sin intervención manual en simulación y validada parcialmente en hardware real.
- Monitoreo remoto en tiempo real del estado completo del sistema (pose, mapa, ruta, liveness de nodos).

---

## Estructura del repositorio

```
robotics_ws/src/
├── puzzlebot_description/      # URDF/Xacro del robot
├── puzzlebot_gazebo/           # Simulación (mundos + modelos)
├── puzzlebot_localisation/     # SLAM propio, Cartographer, MCL, ICP
├── puzzlebot_navigation/       # Costmap, RRT, Pure Pursuit, DWA
├── puzzlebot_vision/           # Detección YOLO, alineación, QR
├── puzzlebot_control/          # Control PID
├── puzzlebot_fpga_controller/  # Puente SPI ↔ FPGA
├── puzzlebot_voice/            # Reconocimiento de voz HMM
├── puzzlebot_gui/              # Dashboard web gRPC
└── puzzlebot_mission/          # Orquestador FSM
```

---

*Proyecto de integración de robótica móvil autónoma — Tecnológico de Monterrey, 2025.*
