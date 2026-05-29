FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /ws

# Herramientas base + rosdep
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-pip \
    python3-numpy \
    python3-transforms3d \
    ros-humble-xacro \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-robot-state-publisher \
    ros-humble-tf2-ros \
    && rm -rf /var/lib/apt/lists/*

# Inicializar rosdep una sola vez en la imagen
RUN rosdep init || true && rosdep update

# Copiar solo los package.xml primero para cachear el paso de rosdep
COPY src/puzzlebot_challenge/package.xml     src/puzzlebot_challenge/package.xml
COPY src/puzzlebot_control/package.xml       src/puzzlebot_control/package.xml
COPY src/puzzlebot_description/package.xml   src/puzzlebot_description/package.xml
COPY src/puzzlebot_gazebo/package.xml        src/puzzlebot_gazebo/package.xml
COPY src/puzzlebot_localisation/package.xml  src/puzzlebot_localisation/package.xml

RUN apt-get update && \
    . /opt/ros/humble/setup.sh && \
    rosdep install --from-paths src --ignore-src -r -y && \
    rm -rf /var/lib/apt/lists/*

# Copiar el resto del workspace
COPY src/ src/

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY docker/ci.sh /docker/ci.sh
COPY docker/demo.sh /docker/demo.sh
RUN chmod +x /docker/ci.sh /docker/demo.sh

ENTRYPOINT ["/entrypoint.sh"]
