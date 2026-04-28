#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool
import math


def generate_square(side):
    return [
        (side, 0.0),
        (side, side),
        (0.0,  side),
        (0.0,  0.0),
    ]


def generate_pentagon(side):
    n = 5
    # Radio del círculo que contiene el pentágono
    r = side / (2.0 * math.sin(math.pi / n))

    points = []
    for i in range(n):
        # Ángulo de cada vértice 
        angle = math.pi / 2.0 + (2.0 * math.pi * i / n)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        points.append((round(x, 4), round(y, 4)))

    return points


class SetpointGenerator(Node):

    def __init__(self):
        super().__init__('setpoint_generator')

        # Parametros 
        self.declare_parameter('trajectory',  'square')
        self.declare_parameter('side_length', 1.0)
        self.declare_parameter('loop',        False)

        trajectory  = self.get_parameter('trajectory').value
        side_length = self.get_parameter('side_length').value
        self.loop   = self.get_parameter('loop').value

        # Lista de puntos
        if trajectory == 'square':
            self.waypoints = generate_square(side_length)
        elif trajectory == 'pentagon':
            self.waypoints = generate_pentagon(side_length)
        else:
            self.get_logger().warn(
                f'Trayectoria "{trajectory}" no reconocida. Usando cuadrado.'
            )
            self.waypoints = generate_square(side_length)

        # Índice del waypoint actual
        self.current_index = 0

        # Flag para saber si el controlador ya llegó al goal actual.
        self.goal_reached = True

        # Bandera para saber si terminamos todos los puntos
        self.finished = False

        # Suscripcion a goal_reached
        self.create_subscription(Bool, 'goal_reached', self.goal_reached_callback, 10)

        # Publicador de set_point
        self.setpoint_pub = self.create_publisher(Vector3, 'set_point', 10)

        # Timer - publica el setpoint actual a 10 Hz
        self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(
            f'Setpoint generator listo | '
            f'trayectoria={trajectory} | lado={side_length} m | '
            f'puntos={self.waypoints} | loop={self.loop}'
        )

    def goal_reached_callback(self, msg: Bool):
        if msg.data:
            self.goal_reached = True

    def timer_callback(self):
        # Si ya terminamos todos los puntos
        if self.finished:
            return

        # Si el controlador llegó al goal actual → avanzar al siguiente
        if self.goal_reached:
            self.goal_reached = False   # resetear flag

            # Avanzar índice
            # Si ya terminamos la lista:
            if self.current_index >= len(self.waypoints):
                if self.loop:
                    # Reiniciar desde el principio
                    self.current_index = 0
                    self.get_logger().info('Trayectoria completada — reiniciando...')
                else:
                    # Terminar
                    self.finished = True
                    self.get_logger().info(
                        'Trayectoria completada. Todos los puntos alcanzados.'
                    )
                    return

            # Publicar el siguiente waypoint
            xg, yg = self.waypoints[self.current_index]

            msg   = Vector3()
            msg.x = float(xg)
            msg.y = float(yg)
            msg.z = 0.0
            self.setpoint_pub.publish(msg)

            self.get_logger().info(
                f'Publicando setpoint {self.current_index + 1}/'
                f'{len(self.waypoints)}: ({xg}, {yg})'
            )

            # Avanzar para la próxima vez
            self.current_index += 1

        else:
            # Mientras el robot no ha llegado, seguir publicando el mismo goal
            if self.current_index > 0:
                xg, yg = self.waypoints[self.current_index - 1]
                msg    = Vector3()
                msg.x  = float(xg)
                msg.y  = float(yg)
                msg.z  = 0.0
                self.setpoint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SetpointGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()