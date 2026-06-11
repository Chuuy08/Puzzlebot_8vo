# PuzzleBot Autonomous Logistics Platform

**Sistema robótico autónomo para automatización logística de pallets basado en ROS 2**

> Plataforma móvil diferencial (PuzzleBot, Manchester Robotics) con percepción visual, localización probabilística, planeación y seguimiento de trayectorias, control de actuadores vía FPGA, comandos de voz y panel de monitoreo web — integrados bajo una arquitectura distribuida de nodos ROS 2.

---

## Tabla de contenidos

1. [Introducción](#1-introducción)
2. [Arquitectura general del sistema](#2-arquitectura-general-del-sistema)
3. [Descripción de módulos](#3-descripción-de-módulos)
4. [Fundamentos matemáticos y algoritmos](#4-fundamentos-matemáticos-y-algoritmos)
5. [Lógica de comportamiento del robot](#5-lógica-de-comportamiento-del-robot)
6. [Tecnologías utilizadas](#6-tecnologías-utilizadas)
7. [Ejecución del sistema](#7-ejecución-del-sistema)
8. [Resultados](#8-resultados)
9. [Limitaciones](#9-limitaciones)
10. [Trabajo futuro](#10-trabajo-futuro)

---

## 1. Introducción

### 1.1 Descripción general

Este proyecto implementa un **sistema robótico móvil autónomo completo** sobre la plataforma **PuzzleBot** (robot diferencial de dos ruedas con Jetson Nano embebida), orientado a tareas de **automatización logística en almacén**: identificación, alineación y manipulación de pallets en estaciones de recolección (rodillos, racks) y su posterior entrega en una zona designada.

El sistema integra, de extremo a extremo, las capas típicas de un robot autónomo moderno:

- **Percepción**: cámara RGB + LiDAR 2D, detección de pallets y códigos QR mediante un modelo YOLO entrenado.
- **Localización**: filtro de partículas (Monte Carlo Localization) fusionando odometría y LiDAR sobre un mapa de ocupación pre-construido.
- **Navegación**: planeación global (RRT bidireccional), seguimiento de trayectoria (Pure Pursuit) y evasión local de obstáculos (DWA).
- **Control de bajo nivel**: lazos PID de velocidad lineal/angular para el seguimiento de waypoints.
- **Actuación mecánica**: subsistema de elevación tipo montacargas (fork lift) controlado por una **FPGA** vía **SPI**, comandado desde ROS 2.
- **Interacción humana**: reconocimiento de comandos de voz (HMM) y un **dashboard web** (gRPC + protobuf) para monitoreo en tiempo real.
- **Orquestación**: una máquina de estados finita (mission_manager) que coordina todos los subsistemas durante una misión completa.

### 1.2 Problema que resuelve

En entornos logísticos pequeños/medianos (almacenes, líneas de producción, laboratorios), el movimiento de pallets entre estaciones de carga (rodillos transportadores, racks de almacenamiento) y zonas de entrega es una tarea repetitiva, sensible a errores de posicionamiento y costosa en mano de obra. El sistema propuesto aborda este problema mediante un robot que:

1. Se localiza de forma autónoma dentro de un mapa conocido del almacén.
2. Navega de forma segura entre zonas de interés evitando obstáculos dinámicos.
3. Detecta visualmente la presencia de un pallet y, si aplica, lee un código QR para identificar su contenido/destino.
4. Se alinea y aproxima de forma precisa al pallet.
5. Acciona el mecanismo de elevación (FPGA) para tomar o depositar la carga.
6. Repite el ciclo hasta completar la misión, reportando su estado en todo momento a un operador humano vía GUI o comandos de voz.

Este flujo reproduce, a escala reducida, el comportamiento de un AGV (*Automated Guided Vehicle*) o AMR (*Autonomous Mobile Robot*) industrial.

---

## 2. Arquitectura general del sistema

### 2.1 Visión por capas

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CAPA DE INTERFAZ                              │
│   puzzlebot_gui (gRPC/Web Dashboard)   |   puzzlebot_voice (HMM)      │
└───────────────▲────────────────────────────────────▲─────────────────┘
                 │ telemetría                         │ /voice_cmd
┌────────────────┴────────────────────────────────────┴─────────────────┐
│                     CAPA DE MISIÓN / DECISIÓN                         │
│              puzzlebot_mission · mission_manager_node (FSM)          │
│  - Orquesta navegación, alineación, FPGA y reporta estado            │
└───┬───────────────────────┬──────────────────────────┬────────────────┘
    │ /goal_pose             │ /align_and_approach/*    │ /deteccion_pallet
    │ /cancel_navigation     │ /pallet_detected, QR     │ /alineation/booleano
┌───▼───────────────┐  ┌─────▼──────────────────┐  ┌────▼────────────────┐
│   NAVEGACIÓN      │  │       PERCEPCIÓN        │  │   ACTUACIÓN FPGA    │
│ rrt_node          │  │ align_and_approach.py   │  │ fpga_controller_node│
│ path_follower_node│  │ tracking.py (YOLO)      │  │ (SPI → FSM en FPGA) │
│ dwa_node          │  │ camera_inference.py     │  └─────────────────────┘
│ costmap_node      │  └─────────────────────────┘
└───┬───────────────┘
    │ /cmd_vel_reference, /cmd_vel
┌───▼───────────────────────────────────────────────────────────────────┐
│                       CAPA DE LOCALIZACIÓN                            │
│   mcl_node (Monte Carlo Localization) · active_localization_node     │
│   /odom + /scan + /map  →  /mcl_pose, /particle_cloud, map→odom TF   │
└───┬─────────────────────────────────────────────────────────────────┬─┘
    │ /odom, /scan                                                     │ /cmd_vel
┌───▼──────────────────────────────────────┐               ┌───────────▼────────┐
│            CAPA DE HARDWARE               │               │  CONTROL PID       │
│  Encoders · IMU · LiDAR · Cámara · Motores│◄──────────────┤  control.py        │
│  Microcontrolador Jetson Nano + FPGA      │               │  (Kp/Ki/Kd)        │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Flujo de datos end-to-end

1. **Sensado**: los encoders de las ruedas generan `/odom` (Odometry); el LiDAR publica `/scan` (LaserScan); la cámara publica `/video_source/compressed` (CompressedImage).
2. **Localización**: `mcl_node` fusiona `/odom` + `/scan` contra el mapa estático `/map` mediante un filtro de partículas, publicando `/mcl_pose` (PoseWithCovarianceStamped) y la transformación `map → odom`. Si la confianza cae (`/mcl_converged = false`), `active_localization_node` ejecuta una rutina de "wandering" para recuperar la convergencia.
3. **Construcción del costmap**: `costmap_node` combina el mapa estático inflado con obstáculos dinámicos detectados por el LiDAR, publicando `/costmap` (OccupancyGrid).
4. **Planeación global**: cuando `mission_manager_node` publica un `/goal_pose`, `rrt_node` calcula una ruta libre de colisiones sobre el `/costmap` mediante **Bidirectional RRT** y publica `/global_path` (nav_msgs/Path).
5. **Seguimiento de trayectoria**: `path_follower_node` ejecuta **Pure Pursuit** sobre `/global_path` y `/mcl_pose`, generando `/cmd_vel_reference` y reportando `/waypoint_reached`.
6. **Evasión local**: `dwa_node` recibe `/cmd_vel_reference` y `/scan`, y mediante **Dynamic Window Approach** genera el `/cmd_vel` final, ajustado para evitar colisiones inminentes.
7. **Percepción de pallets**: `align_and_approach.py` procesa `/video_source/compressed` con un modelo **YOLO**, detecta la clase `pallet`, localiza un **código QR** dentro del bounding box, y ejecuta una FSM de alineación (ALIGNING → APPROACHING → BLIND) que publica `/cmd_vel`, `/pallet_detected`, `/pallet_has_qr`, `/pallet_qr_content` y `/alineation/booleano`.
8. **Decisión de misión**: `mission_manager_node` consume todas las señales anteriores y conduce una **máquina de estados finita** que decide a dónde navegar, cuándo activar la alineación visual, cuándo invocar al FPGA y cuándo dar por completada la misión.
9. **Actuación mecánica**: `fpga_controller_node` recibe `/deteccion_pallet` (RACK/RODILLO) y `/alineation/booleano`, y mediante una librería SPI (`libspi_fsm.so`) envía comandos a una **FSM implementada en FPGA** que controla los actuadores del mecanismo elevador (fork lift), con secuencias de tiempo calibradas por tipo de estación.
10. **Interfaz humana**: `puzzlebot_voice` reconoce comandos hablados (adelante, atrás, izquierda, derecha, parar, levantar, soltar, etc.) y los traduce a `/voice_cmd`; `puzzlebot_gui` agrega telemetría de todos los nodos (pose, mapa, costmap, ruta, particle cloud, video) y la transmite vía **gRPC/Protobuf** a un dashboard web.

### 2.3 Interacción entre nodos ROS 2

El sistema sigue el patrón **publish/subscribe desacoplado** característico de ROS 2: cada nodo es independiente y se comunica exclusivamente mediante tópicos, con `mission_manager_node` actuando como **orquestador central** que no procesa señales crudas, sino las salidas semánticas de cada subsistema (pose estimada, ruta calculada, detección de pallet, etc.). Adicionalmente, `mission_manager_node` utiliza **servicios de parámetros dinámicos** (`get_parameters`/`set_parameters`) para reconfigurar en caliente el radio de inflación del costmap y el radio del robot en `dwa_node` durante maniobras de aproximación fina a los pallets, donde se requiere mayor tolerancia a la cercanía de obstáculos.

---

## 3. Descripción de módulos

### 3.1 Visión (`puzzlebot_vision`)

- **`align_and_approach.py`**: nodo principal de percepción para la fase de manipulación. Implementa una FSM de tres fases:
  - **ALIGNING**: detecta el bounding box de la clase `pallet` (YOLO), calcula el error horizontal entre el centro del bbox y el centro de la imagen, y aplica pulsos angulares proporcionales (`ALIGN_PULSE_ANGULAR`, `ALIGN_PULSE_GAIN`) hasta que el error cae por debajo de `ALIGN_ERROR_THRESHOLD`.
  - **APPROACHING**: una vez alineado, controla la velocidad lineal proporcionalmente al área del bounding box (`Kp_approach_linear`) hasta alcanzar `TARGET_AREA_RATIO`, corrigiendo desviaciones laterales con `Kp_approach_correction`.
  - **BLIND**: tramo final de aproximación ciego (sin retroalimentación visual, por oclusión cercana), controlado por odometría pura hasta `BLIND_APPROACH_DISTANCE`.
  - Detecta además códigos QR (clases YOLO 4/5) dentro de la región del pallet y decodifica su contenido para identificar el tipo de carga.
- **`tracking.py`**: variante de seguimiento continuo basada en YOLO con tracker (re-identificación de la misma instancia de pallet entre frames).
- **`evaluate_model.py`**: script de evaluación de métricas del modelo (precisión, recall, mAP) sobre datasets etiquetados.
- **`camera_inference.py`**: utilidad de inferencia en vivo (webcam o `/video_source/compressed`) para depuración del modelo.

### 3.2 Navegación (`puzzlebot_navigation`)

- **`costmap_node.py`**: genera el mapa de costos combinando el mapa estático (`/map`) inflado por `inflation_radius` con obstáculos dinámicos del `/scan`, produciendo `/costmap` (0 = libre, 99 = zona inflada, 100 = obstáculo).
- **`rrt_node.py`**: planeador global mediante **RRT bidireccional** (BiRRT) con muestreo sesgado al objetivo (`goal_bias`), suavizado de ruta opcional (`smooth_path`) y límite de iteraciones (`max_iterations`).
- **`path_follower_node.py`**: controlador de seguimiento **Pure Pursuit**, con una FSM de tres estados (IDLE → ALIGN → DRIVE → DONE): primero orienta al robot hacia el primer punto de la ruta (si el error angular supera `align_threshold_deg`), luego avanza siguiendo el punto de mira a `lookahead_distance`.
- **`dwa_node.py`**: capa de evasión reactiva **Dynamic Window Approach**, que muestrea trayectorias candidatas `(v, ω)` dentro de la ventana dinámica del robot y las puntúa según *heading* hacia el objetivo, *clearance* respecto a obstáculos del `/scan`, y velocidad, combinando los términos con pesos `w_heading`, `w_clearance`, `w_velocity`.

### 3.3 Localización (`puzzlebot_localisation`)

- **`mcl_node.py`**: implementación de **Monte Carlo Localization (filtro de partículas)** con 1000–1500 partículas. Soporta dos modos: *tracking* (convergido, ruido bajo `sigma_hit`) y *global* (re-localización, ruido alto `sigma_hit_global`, menor densidad de rayos `beam_step`). Publica la pose estimada `/mcl_pose`, la nube de partículas `/particle_cloud` y la transformación `map → odom`.
- **`active_localization_node.py`**: rutina de recuperación activa — cuando `/mcl_converged = false`, ejecuta movimientos de "wandering" (`/cmd_vel`, `/mcl_wandering = true`) para generar diversidad de observaciones y forzar la reconvergencia del filtro, cancelando temporalmente la navegación (`/cancel_navigation`).
- **`occupancy_grid_node.py`**: construcción de mapas de ocupación a partir de `/odom` + `/scan` (utilidad de mapeo).
- **`icp_node.py`**: alternativa de localización/refinamiento mediante **Iterative Closest Point** (alineación de nubes de puntos del LiDAR contra el mapa).

### 3.4 Control (`puzzlebot_control`)

- **`control.py`**: controlador de bajo nivel basado en **dos lazos PID en cascada** (distancia y orientación) que recibe un punto objetivo (`/set_point`, Vector3) y la odometría (`/odom`), calculando velocidades lineal/angular saturadas (`v_max`, `w_max`) hasta alcanzar el objetivo dentro de un `threshold`, reportando `/goal_reached`.

### 3.5 FPGA (`puzzlebot_fpga_controller`)

- **`fpga_controller_node.py`**: puente ROS 2 → **SPI** → **FPGA**, mediante una librería compartida en C (`libspi_fsm.so`) que implementa el protocolo de comunicación con una **máquina de estados finita sintetizada en hardware** (`fpga/control_fpga_fsm/`).
  - Recibe `/deteccion_pallet` (`"RACK"` o `"RODILLO"`), `/alineation/booleano` y `/waypoint_reached`.
  - Traduce estos eventos a comandos SPI (`CMD_RACK = 0x10`, `CMD_RODILLO = 0x11`, `CMD_ALINEADO = 0x21`, `CMD_WAYPOINT = 0x22`) que la FPGA interpreta en sus propios estados (`STATE_IDLE`, `STATE_ESPERAR_ALINEACION`, `STATE_ESPERAR_WAYPOINT`).
  - Ejecuta secuencias de tiempo calibradas para el mecanismo de elevación según el tipo de estación (p. ej. RACK: `bajar_vision = 1700 ms`; RODILLO: `subir = 1500 ms`, `bajar_vision = 7200 ms`).
  - **Nota de despliegue**: este nodo corre nativamente en la **Jetson del robot (ARM)**; la copia presente en este workspace x86 es de referencia/desarrollo.

### 3.6 Comandos de voz (`puzzlebot_voice`)

- **`voice_cmd.py`**: nodo de reconocimiento de comandos basado en **Modelos Ocultos de Markov (HMM)**, entrenado sobre un dataset de audio propio (comandos: *adelante, atrás, izquierda, derecha, parar, levantar, bajar, iniciar*).
- **`voice_utils.py`**: pipeline de procesamiento de señal — pre-énfasis, ventaneo, autocorrelación, extracción de coeficientes **LPC/LSF** como características acústicas.
- **`hmm_utils.py`**: entrenamiento (Baum-Welch) e inferencia (Viterbi/forward) de los modelos HMM por palabra.
- Salida: `/voice_cmd` (String) — puede integrarse como entrada alternativa de teleoperación o disparador de eventos de misión.

### 3.7 Interfaz gráfica (`puzzlebot_gui`)

- **`ros_bridge.py`**: nodo ROS 2 que se suscribe a la totalidad de los tópicos de telemetría (`/odom`, `/mcl_pose`, `/map`, `/costmap`, `/global_path`, `/particle_cloud`, `/cmd_vel`, `/cmd_vel_reference`, `/mcl_converged`, `/mcl_wandering`, `/joint_states`, video de alineación) y los serializa a mensajes **Protobuf**.
- **`grpc_server.py`**: expone un servidor **gRPC** que sirve estos datos a un cliente web.
- **`web/`**: frontend del dashboard — visualiza mapa, partículas del MCL, ruta global, pose del robot, estado de los nodos (heartbeat/liveness) y stream de la cámara con detecciones anotadas.
- Funcionalidad de **liveness tracking**: cada subsistema (mcl_node, rrt_node, dwa_node, path_follower_node, costmap_node, active_localization_node, joint_state_publisher) reporta su última actualización, permitiendo detectar nodos caídos desde la GUI.

### 3.8 Planeador de misión (`puzzlebot_mission`)

- **`mission_manager_node.py`**: el **cerebro del sistema**. Implementa una FSM con los siguientes estados principales:

  `WAITING_LOCALIZATION → NAV_TO_AREA_GENERAL → NAV_TO_SWEEP_PHASE → SWEEP_SAMPLING → NAV_TO_PALLET → WAITING_ALIGNMENT → NAV_TO_DELIVERY → REVERSE_TO_ACCESO → MISSION_COMPLETE / ABORTED`

  - Carga waypoints predefinidos de un YAML (`waypoints_yaml_path`) para cada zona (rodillos, racks 1–3, zona de entrega).
  - Coordina la navegación (publica `/goal_pose`, escucha `/waypoint_reached`).
  - Durante `SWEEP_SAMPLING`, gira el robot en un rango angular (`sweep_range_deg`) para maximizar la probabilidad de detección visual del pallet, tomando `sweep_samples_per_stop` muestras por parada.
  - Activa `align_and_approach` (`/align_and_approach/active`) y espera `/alineation/booleano` con un timeout (`nav_timeout_s`).
  - Reconfigura dinámicamente `costmap_node` y `dwa_node` (radios de inflación reducidos) para permitir aproximaciones cercanas durante la manipulación.
  - Notifica a `fpga_controller_node` el tipo de estación detectada (`/deteccion_pallet`) y espera `fpga_settle_time_s` para la finalización del ciclo mecánico.
  - Ejecuta una maniobra de retroceso controlado (`reverse_speed`) tras completar la entrega.

---

## 4. Fundamentos matemáticos y algoritmos

### 4.1 Control PID

El controlador de bajo nivel (`puzzlebot_control/control.py`) y los pulsos de alineación visual emplean la forma discreta del controlador **Proporcional-Integral-Derivativo**:

```
u(k) = Kp·e(k) + Ki·Σ e(i)·Δt + Kd·(e(k) - e(k-1)) / Δt
```

donde `e(k)` es el error en el instante `k` (distancia euclidiana al objetivo o error angular de heading). El sistema implementa **dos lazos PID desacoplados**:

- **Lazo de distancia**: `e_d = √((x_g - x)² + (y_g - y)²)` → controla `v` (velocidad lineal), saturada a `v_max`.
- **Lazo de orientación**: `e_θ = atan2(y_g - y, x_g - x) - θ` (normalizado a [-π, π]) → controla `ω` (velocidad angular), saturada a `w_max`.

En la práctica, las ganancias dominantes son las proporcionales (`Kp_d = 0.3`, `Kp_θ = 0.6`), con términos integrales/derivativos típicamente nulos o muy pequeños, priorizando estabilidad sobre velocidad de respuesta — adecuado para un robot de masa baja con actuación discreta (PWM vía FPGA/microcontrolador).

### 4.2 Estimación de pose — Monte Carlo Localization

El `mcl_node` implementa el algoritmo clásico de **filtro de partículas** para localización:

1. **Predicción (modelo de movimiento odométrico)**: cada partícula `i` se propaga según el modelo de odometría con ruido gaussiano parametrizado por `alpha1..alpha4`:

```
δ_rot1  = atan2(Δy, Δx) - θ_t-1
δ_trans = √(Δx² + Δy²)
δ_rot2  = θ_t - θ_t-1 - δ_rot1

δ_rot1' = δ_rot1  - 𝒩(0, α1·δ_rot1² + α2·δ_trans²)
δ_trans' = δ_trans - 𝒩(0, α3·δ_trans² + α4·(δ_rot1² + δ_rot2²))
δ_rot2'  = δ_rot2  - 𝒩(0, α1·δ_rot2² + α2·δ_trans²)
```

2. **Actualización (modelo de sensor — beam model simplificado)**: para cada partícula, se hace *ray-casting* simulado sobre el mapa de ocupación para un subconjunto de haces del LiDAR (controlado por `beam_step`), comparando la distancia esperada con la medida real `/scan`. La verosimilitud de cada haz se modela como una mezcla gaussiana centrada en la lectura esperada:

```
p(z | x) ∝ z_hit · exp( -(z_meas - z_expected)² / (2·σ_hit²) )
```

3. **Remuestreo (Low-Variance Resampling)**: las partículas se remuestrean proporcionalmente a su peso `w_i`, concentrando la población en regiones de alta verosimilitud.

4. **Estimación de pose**: la pose publicada en `/mcl_pose` corresponde a la media ponderada (o partícula de máxima verosimilitud) de la nube, junto con su matriz de covarianza 3×3 (x, y, θ).

5. **Modo dual (tracking/global)**: cuando `/mcl_converged = false` (covarianza alta o detección de "secuestro"), el filtro cambia a parámetros de búsqueda global (`sigma_hit_global` mayor, menor `beam_step` → menor costo computacional por partícula, mayor diversidad).

### 4.3 Transformaciones de marcos de referencia (TF)

El sistema mantiene el árbol de transformadas estándar de ROS 2 para navegación móvil:

```
map → odom → base_link → {laser_frame, camera_link}
```

- **`map → odom`**: publicada por `mcl_node`; representa la corrección acumulada de deriva odométrica. Se calcula como:

```
T_map_odom = T_map_base · (T_odom_base)⁻¹
```

- **`odom → base_link`**: integración de la odometría de las ruedas (encoders), sujeta a deriva no acotada.
- **`base_link → laser_frame / camera_link`**: transformaciones estáticas definidas en el URDF (`puzzlebot_description`), con offsets configurables (`laser_x_offset`, `laser_angle_offset` — este último compensa el montaje rotado 180° del LiDAR en el robot real frente a la simulación).

Todos los algoritmos de planeación y control operan en el marco `map` (global), mientras que la odometría cruda y el LiDAR llegan en marcos locales (`odom`, `laser_frame`) y se transforman mediante `tf2`.

### 4.4 Procesamiento de imagen y detección

- **Pipeline de visión**: `CompressedImage (/video_source/compressed)` → decodificación JPEG (OpenCV) → inferencia **YOLO** → bounding boxes con clase y confianza.
- **Clases relevantes**: `pallet` (clase 3), `qr` / `qr-code` (clases 4, 5).
- **Decodificación QR**: sobre la región de interés (ROI) del bounding box de tipo QR, se aplica un decodificador (p. ej. `pyzbar`/OpenCV `QRCodeDetector`) para extraer el contenido textual (`/pallet_qr_content`).
- **Cálculo de error de alineación**: dado el bounding box `(x_min, x_max)` y el ancho de imagen `W`:

```
error_x = ( (x_min + x_max)/2 - W/2 ) / (W/2)     ∈ [-1, 1]
```

Si `|error_x| > ALIGN_ERROR_THRESHOLD`, se emite un pulso angular `ω = -sign(error_x) · ALIGN_PULSE_ANGULAR · ALIGN_PULSE_GAIN`.

- **Cálculo de error de aproximación**: se usa el **ratio de área del bounding box** respecto al área total de la imagen como proxy de distancia:

```
area_ratio = (bbox_w · bbox_h) / (W · H)
v = Kp_approach_linear · (TARGET_AREA_RATIO - area_ratio)
```

### 4.5 Planeación de trayectoria (conceptual)

- **RRT Bidireccional (BiRRT)**: dos árboles crecen simultáneamente desde el inicio y la meta sobre el espacio libre del `/costmap`. En cada iteración:
  1. Se muestrea un punto aleatorio en el espacio (con probabilidad `goal_bias` se muestrea directamente la meta).
  2. Se extiende el árbol más cercano hacia ese punto una distancia `step_size`, verificando colisión contra el costmap.
  3. Se intenta conectar ambos árboles; si se logra, se reconstruye la ruta y opcionalmente se aplica suavizado (`smooth_path`) — eliminación de nodos colineales y *shortcutting* mediante chequeos de línea de vista.

- **Pure Pursuit (seguimiento)**: dado un punto de mira (*lookahead point*) sobre `/global_path` a distancia `lookahead_distance` de la pose actual, se calcula la curvatura requerida:

```
κ = 2·sin(α) / L_d
ω = v · κ
```

donde `α` es el ángulo entre la orientación del robot y la línea hacia el punto de mira, y `L_d` es la distancia de lookahead.

- **Dynamic Window Approach (DWA)**: genera un conjunto de pares `(v, ω)` admisibles dentro de la ventana dinámica (limitada por aceleraciones máximas), simula trayectorias a `sim_time` segundos, y selecciona la de mayor puntuación:

```
score(v, ω) = w_heading · heading(v,ω) + w_clearance · clearance(v,ω) + w_velocity · velocity(v,ω)
```

- `heading`: alineación de la trayectoria simulada con la dirección al siguiente punto del `/global_path`.
- `clearance`: distancia mínima a obstáculos detectados en `/scan` a lo largo de la trayectoria.
- `velocity`: preferencia por velocidades lineales mayores (evita estancamiento).

---

## 5. Lógica de comportamiento del robot

### 5.1 Máquina de estados de la misión

```
                ┌────────────────────────┐
                │  WAITING_LOCALIZATION   │  ← espera /mcl_converged = true
                └───────────┬─────────────┘
                             │
                ┌────────────▼─────────────┐
                │   NAV_TO_AREA_GENERAL     │  ← /goal_pose hacia zona (rodillos/rack)
                └───────────┬─────────────┘
                             │ /waypoint_reached
                ┌────────────▼─────────────┐
                │   NAV_TO_SWEEP_PHASE      │
                └───────────┬─────────────┘
                             │
                ┌────────────▼─────────────┐
                │     SWEEP_SAMPLING        │  ← gira ±sweep_range_deg buscando pallet
                └───────────┬─────────────┘
                  /pallet_detected = true
                ┌────────────▼─────────────┐
                │      NAV_TO_PALLET        │  ← navega hacia el pallet detectado
                └───────────┬─────────────┘
                             │
                ┌────────────▼─────────────┐
                │    WAITING_ALIGNMENT      │  ← activa align_and_approach
                │  (ALIGNING→APPROACHING→   │  ← /pallet_has_qr, /pallet_qr_content
                │       BLIND)              │  ← /alineation/booleano = true
                └───────────┬─────────────┘
                             │ → /deteccion_pallet (RACK|RODILLO) → FPGA (lift)
                ┌────────────▼─────────────┐
                │     NAV_TO_DELIVERY       │  ← navega a zona de entrega
                └───────────┬─────────────┘
                             │ → FPGA (lower)
                ┌────────────▼─────────────┐
                │    REVERSE_TO_ACCESO      │  ← retrocede a velocidad reverse_speed
                └───────────┬─────────────┘
                             │
                ┌────────────▼─────────────┐
                │     MISSION_COMPLETE      │
                └────────────────────────────┘

         (cualquier estado) ──timeout / fallo──► ABORTED
```

### 5.2 Ejemplo de flujo: detectar pallet → validar QR → alinearse → ejecutar acción

1. **Detección**: durante `SWEEP_SAMPLING`, `align_and_approach` analiza cada frame de `/video_source/compressed` con YOLO. Al encontrar la clase `pallet` con confianza suficiente, publica `/pallet_detected = true`.
2. **Validación QR**: dentro del bounding box del pallet, se buscan detecciones de clase `qr`/`qr-code`. Si se encuentra, se decodifica y se publica `/pallet_qr_content` (string) y `/pallet_has_qr = true`. `mission_manager_node` usa este contenido para decidir el destino final de la carga.
3. **Alineación**: `mission_manager_node` transiciona a `WAITING_ALIGNMENT` y activa `/align_and_approach/active = true`. El nodo de visión ejecuta:
   - **ALIGNING**: pulsos angulares hasta `|error_x| < ALIGN_ERROR_THRESHOLD`.
   - **APPROACHING**: avance proporcional al área del bbox hasta `area_ratio ≥ TARGET_AREA_RATIO`.
   - **BLIND**: tramo final por odometría (`BLIND_APPROACH_DISTANCE`) cuando el pallet sale del campo de visión por proximidad.
   - Al completar, publica `/alineation/booleano = true`.
4. **Ejecución de acción (FPGA)**: `mission_manager_node` publica `/deteccion_pallet` (`"RACK"` o `"RODILLO"` según la zona/QR). `fpga_controller_node` traduce esto a comandos SPI (`CMD_RACK`/`CMD_RODILLO`), la FPGA ejecuta la secuencia de elevación calibrada (subir → sostener → bajar) y, al recibir `/alineation/booleano`, confirma con `CMD_ALINEADO`.
5. **Continuación**: tras `fpga_settle_time_s`, la misión transiciona a `NAV_TO_DELIVERY`, repitiendo el ciclo de navegación hacia la zona de entrega, donde se ejecuta la secuencia inversa (depósito de la carga).

---

## 6. Tecnologías utilizadas

| Categoría | Tecnologías |
|---|---|
| **Middleware robótico** | ROS 2 (rclpy), tf2, nav_msgs, sensor_msgs, geometry_msgs |
| **Lenguajes** | Python 3 (nodos ROS 2, visión, voz, GUI), C/C++ (driver SPI `libspi_fsm.so`), VHDL/Verilog (FSM en FPGA) |
| **Visión por computadora** | OpenCV, YOLO (Ultralytics/Darknet), pyzbar/QRCodeDetector |
| **Localización / SLAM** | Filtro de partículas (MCL) propio, ICP, alternativas: Cartographer, AMCL, SLAM Toolbox |
| **Planeación y control** | RRT bidireccional, Pure Pursuit, Dynamic Window Approach (DWA), PID |
| **Hardware embebido** | NVIDIA Jetson Nano, FPGA (control SPI del mecanismo elevador) |
| **Procesamiento de audio / Voz** | HMM (Baum-Welch / Viterbi), extracción LPC/LSF, autocorrelación |
| **Simulación** | Gazebo (mundos `.sdf`, modelos URDF/Xacro) |
| **Interfaz / Telemetría** | gRPC, Protocol Buffers, dashboard web (HTML/JS) |
| **Build system** | colcon, ament_python / ament_cmake |

---

## 7. Ejecución del sistema

> Los siguientes comandos asumen un workspace ROS 2 ya compilado (`colcon build`) y `source install/setup.bash` ejecutado.

### 7.1 Simulación completa (Gazebo + MCL + Navegación + Misión)

```bash
# 1. Levantar la simulación (mundo + robot + sensores)
ros2 launch puzzlebot_gazebo gazebo_world_launch.py world:=dt_world.sdf
ros2 launch puzzlebot_gazebo gazebo_puzzlebot_launch.py

# 2. Localización (MCL) — incluye map_server + lifecycle_manager
ros2 launch puzzlebot_localisation mcl_launch.py

# 3. Pila de navegación (costmap, RRT, path follower, DWA)
ros2 launch puzzlebot_navigation navigation_launch.py

# 4. Percepción / alineación a pallets
ros2 run puzzlebot_vision align_and_approach

# 5. Orquestador de misión
ros2 launch puzzlebot_mission mission_launch.py
```

### 7.2 Robot real — bring-up completo

```bash
ros2 launch puzzlebot_mission full_mission_real_launch.py
```

Este launch file orquesta, con retardos escalonados para asegurar el orden de inicialización:

1. `mcl_real_launch.py` (MCL con `/scan_fixed` y `laser_angle_offset = π`) — t = 0 s
2. `navigation_real_launch.py` (costmap, RRT, path follower, DWA) — t = 6 s
3. `align_and_approach` (percepción/alineación) — t = 6 s
4. `mission_manager_node` — t = 15 s

### 7.3 Subsistemas auxiliares

```bash
# Reconocimiento de voz
ros2 run puzzlebot_voice voice_cmd

# Bridge GUI (gRPC) + dashboard web
ros2 run puzzlebot_gui ros_bridge
ros2 run puzzlebot_gui grpc_server
# Dashboard: abrir puzzlebot_gui/web/index.html (o servidor estático configurado)

# Controlador FPGA (corre en la Jetson del robot, arquitectura ARM)
ros2 run puzzlebot_fpga_controller fpga_controller_node
```

### 7.4 Tópicos clave para depuración

| Tópico | Tipo | Descripción |
|---|---|---|
| `/mcl_pose` | `geometry_msgs/PoseWithCovarianceStamped` | Pose estimada por el filtro de partículas |
| `/particle_cloud` | `geometry_msgs/PoseArray` | Nube de partículas del MCL |
| `/mcl_converged` | `std_msgs/Bool` | Indicador de convergencia de la localización |
| `/costmap` | `nav_msgs/OccupancyGrid` | Mapa de costos (estático + dinámico) |
| `/global_path` | `nav_msgs/Path` | Trayectoria planeada por RRT |
| `/cmd_vel_reference` | `geometry_msgs/Twist` | Comando de velocidad pre-DWA |
| `/cmd_vel` | `geometry_msgs/Twist` | Comando de velocidad final al robot |
| `/pallet_detected`, `/pallet_has_qr`, `/pallet_qr_content` | `std_msgs/Bool`, `std_msgs/String` | Salidas de percepción |
| `/alineation/booleano` | `std_msgs/Bool` | Confirmación de alineación completa |
| `/deteccion_pallet` | `std_msgs/String` | Tipo de estación (`RACK`/`RODILLO`) hacia FPGA |
| `/voice_cmd` | `std_msgs/String` | Comando de voz reconocido |

---

## 8. Resultados

- **Localización robusta**: el filtro MCL converge desde poses iniciales desconocidas (modo global) y mantiene seguimiento estable durante navegación, incluyendo recuperación automática mediante `active_localization_node` ante pérdida de convergencia.
- **Navegación punto a punto**: el robot planea (RRT) y ejecuta (Pure Pursuit + DWA) trayectorias libres de colisión entre zonas del mapa, evitando obstáculos dinámicos no presentes en el mapa estático.
- **Detección y manipulación de pallets**: el pipeline de visión identifica pallets y códigos QR, ejecuta una aproximación en tres fases (visual → visual de cercanía → ciega por odometría) hasta posicionar al robot dentro de tolerancia para la actuación del mecanismo elevador.
- **Actuación coordinada**: la integración ROS 2 ↔ SPI ↔ FPGA permite secuenciar de forma fiable las operaciones mecánicas de carga/descarga en sincronía con los hitos de navegación y alineación.
- **Misión end-to-end**: el `mission_manager_node` ejecuta el ciclo completo (localizar → navegar → buscar → alinear → accionar → entregar → retroceder) sin intervención manual, en simulación y validado parcialmente en hardware real.
- **Monitoreo remoto**: el dashboard web permite observar en tiempo real la pose, el mapa, la nube de partículas, la ruta planeada y el estado de salud (liveness) de cada nodo del sistema.
- **Canal de voz**: comandos hablados básicos son reconocidos de forma confiable bajo el modelo HMM entrenado, habilitando teleoperación o disparo de eventos sin interfaz gráfica.

---

## 9. Limitaciones

- **Dependencia del mapa estático**: la localización (MCL) y la planeación (RRT/costmap) dependen de un mapa de ocupación pre-construido y preciso; cambios estructurales del entorno requieren remapeo manual.
- **Percepción monocular**: la estimación de distancia al pallet se basa en heurísticas de área de bounding box, sensible a variaciones de iluminación, oclusión parcial y orientación del pallet — no hay estimación de profundidad real (sin cámara RGB-D ni fusión con LiDAR en esta etapa).
- **Tramo "ciego" de aproximación**: la fase `BLIND` depende exclusivamente de odometría de corto alcance, acumulando error de deriva si la distancia ciega es mayor a la calibrada.
- **Calibración de tiempos en FPGA**: las secuencias de actuación del mecanismo elevador están temporizadas (no realimentadas por sensores de fin de carrera), lo que las hace sensibles a variaciones de carga, voltaje de batería o desgaste mecánico.
- **Reconocimiento de voz**: el modelo HMM está entrenado sobre un vocabulario y locutores limitados; su robustez ante ruido ambiente de almacén o nuevos hablantes no está garantizada.
- **Validación en hardware real parcial**: varios componentes (FPGA, MCL real, alineación visual con cámara física) han sido probados de forma individual, pero la ejecución de la misión completa end-to-end en el robot físico aún requiere validación de campo extensiva.
- **Arquitectura cruzada (x86/ARM)**: el desarrollo se realiza en x86, mientras que el `fpga_controller_node` y parte del stack real corren exclusivamente en la Jetson (ARM) del robot, lo que introduce fricción en el ciclo de desarrollo/prueba.

---

## 10. Trabajo futuro

1. **Fusión sensorial perceptual**: combinar LiDAR + cámara (o incorporar una cámara de profundidad) para una estimación de distancia al pallet más robusta, reduciendo la dependencia de heurísticas basadas en área de bounding box.
2. **Reemplazo/complemento del LiDAR por visión**: dado que la cámara ya es el sensor principal para manipulación, explorar **localización visual (VSLAM/Visual Odometry)** como respaldo o reemplazo parcial del LiDAR en escenarios donde este no esté disponible.
3. **Retroalimentación de fin de carrera en el mecanismo FPGA**: sustituir las secuencias temporizadas por control en lazo cerrado usando sensores de posición/fin de carrera, mejorando la robustez ante variaciones de carga.
4. **Mejora del modelo de voz**: migrar de HMM a un modelo de reconocimiento de voz más moderno (p. ej. modelos basados en redes neuronales ligeras embebibles en Jetson) para mayor robustez ante ruido y vocabulario extendido.
5. **Planeación con re-planeo dinámico**: incorporar re-planeo incremental (p. ej. RRT* o D* Lite) para adaptar la ruta global ante obstáculos dinámicos persistentes sin reiniciar la planeación completa.
6. **Multi-robot / flotas**: extender la arquitectura de `mission_manager_node` para coordinar múltiples PuzzleBots compartiendo un mapa y asignación dinámica de tareas (zonas de recolección/entrega).
7. **Persistencia y métricas de misión**: registrar logs estructurados (rosbag2) de cada misión para análisis posterior de desempeño (tiempos por fase, tasa de éxito de alineación, precisión de detección QR).
8. **Hardening de la GUI**: autenticación y control de acceso al dashboard web, y panel de control bidireccional (no solo monitoreo) para teleoperación de emergencia.

---

## Estructura del repositorio

```
robotics_ws/
├── src/
│   ├── puzzlebot_description/     # URDF/Xacro del robot
│   ├── puzzlebot_gazebo/          # Mundos y modelos de simulación
│   ├── puzzlebot_localisation/    # MCL, ICP, occupancy grid
│   ├── puzzlebot_navigation/      # Costmap, RRT, Pure Pursuit, DWA
│   ├── puzzlebot_vision/          # YOLO: detección de pallets y QR
│   ├── puzzlebot_voice/           # Reconocimiento de voz (HMM)
│   ├── puzzlebot_control/         # Control PID de bajo nivel
│   ├── puzzlebot_fpga_controller/ # Puente SPI ↔ FPGA (mecanismo elevador)
│   ├── puzzlebot_gui/             # Dashboard web (gRPC + Protobuf)
│   ├── puzzlebot_mission/         # Orquestador de misión (FSM)
│   └── Acts_extras/               # Ejercicios y utilidades de TFs/actuadores
└── README.md
```

---

*Proyecto desarrollado como plataforma de integración de robótica móvil autónoma — percepción, localización, navegación, control y actuación — bajo ROS 2.*
