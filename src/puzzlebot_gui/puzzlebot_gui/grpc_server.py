import asyncio
import logging
import signal
import time

import grpc
import grpc.aio

from .ros_bridge import RosBridge

logger = logging.getLogger(__name__)

# Frecuencias de transmisión
_T_TELEMETRY = 0.10   # 10 Hz
_T_COSTMAP   = 0.50   # 2  Hz
_T_PATH      = 0.10   # 10 Hz
_T_STATUS    = 1.00   # 1  Hz
_T_CAMERA    = 0.067  # ~15 Hz


def _get_pb2():
    try:
        from .proto_gen import puzzlebot_pb2 as pb2
        return pb2
    except ImportError as exc:
        raise RuntimeError(
            'Stubs protobuf no generados. Ejecuta generate_proto.sh primero.'
        ) from exc


def _pb(cls_name, **kwargs):
    return getattr(_get_pb2(), cls_name)(**kwargs)


# ── Constructores de mensajes proto ───────────────────────────────────────────

def _make_telemetry(d):
    return _pb('TelemetryData',
               pos_x=d['pos_x'], pos_y=d['pos_y'], yaw=d['yaw'],
               vel_linear=d['vel_linear'], vel_angular=d['vel_angular'],
               timestamp=d['timestamp'])


def _make_costmap(d):
    pb2 = _get_pb2()
    return pb2.CostmapData(
        data=d['data'], width=d['width'], height=d['height'],
        resolution=d['resolution'],
        origin_x=d['origin_x'], origin_y=d['origin_y'],
        robot_x=d['robot_x'], robot_y=d['robot_y'], robot_yaw=d['robot_yaw'],
        timestamp=d['timestamp'],
        map_data       =d.get('map_data', []),
        map_width      =d.get('map_width', 0),
        map_height     =d.get('map_height', 0),
        map_resolution =d.get('map_resolution', 0.0),
        map_origin_x   =d.get('map_origin_x', 0.0),
        map_origin_y   =d.get('map_origin_y', 0.0),
        map_origin_yaw =d.get('map_origin_yaw', 0.0),
    )


def _make_path(d):
    pb2 = _get_pb2()
    poses     = [pb2.Pose2D(x=p[0], y=p[1], yaw=p[2]) for p in d['poses']]
    particles = [pb2.Pose2D(x=p[0], y=p[1], yaw=p[2]) for p in d.get('particle_poses', [])]
    return pb2.PathData(poses=poses, particle_poses=particles, timestamp=d['timestamp'])


def _make_node_status_list(nodos):
    pb2 = _get_pb2()
    items = [
        pb2.NodeStatus(name=n['name'], alive=n['alive'], last_seen=n['last_seen'])
        for n in nodos
    ]
    return pb2.NodeStatusList(nodes=items, timestamp=time.time())


def _make_camera_frame(f):
    pb2 = _get_pb2()
    dets = [
        pb2.BoundingBox(
            x1=d['x1'], y1=d['y1'], x2=d['x2'], y2=d['y2'],
            label=d['label'], confidence=d['confidence'],
        )
        for d in f.get('detections', [])
    ]
    return pb2.CameraFrame(
        jpeg_data =f.get('jpeg_data', b''),
        detections=dets,
        width     =f.get('width', 0),
        height    =f.get('height', 0),
        timestamp =f.get('timestamp', time.time()),
    )


# ── Servicer gRPC ─────────────────────────────────────────────────────────────

class _Servicer:
    """Implementa los RPCs del .proto usando grpcio async generators."""

    def __init__(self, bridge: RosBridge):
        self._bridge = bridge

    # -- Streams de monitoreo --------------------------------------------------

    async def StreamTelemetry(self, request, context):
        try:
            while True:
                data = self._bridge.get_telemetry()
                if data:
                    yield _make_telemetry(data)
                await asyncio.sleep(_T_TELEMETRY)
        except asyncio.CancelledError:
            pass

    async def StreamCostmap(self, request, context):
        ultimo = -1.0
        try:
            while True:
                data = self._bridge.get_costmap()
                if data and data['timestamp'] != ultimo:
                    ultimo = data['timestamp']
                    yield _make_costmap(data)
                await asyncio.sleep(_T_COSTMAP)
        except asyncio.CancelledError:
            pass

    async def StreamPath(self, request, context):
        ultimo = -1.0
        try:
            while True:
                data = self._bridge.get_path()
                if data and data['timestamp'] != ultimo:
                    ultimo = data['timestamp']
                    yield _make_path(data)
                await asyncio.sleep(_T_PATH)
        except asyncio.CancelledError:
            pass

    async def StreamNodeStatus(self, request, context):
        try:
            while True:
                yield _make_node_status_list(self._bridge.get_node_status())
                await asyncio.sleep(_T_STATUS)
        except asyncio.CancelledError:
            pass

    async def StreamCamera(self, request, context):
        ultimo = -1.0
        try:
            while True:
                frame = self._bridge.get_camera_frame()
                if frame and frame.get('timestamp', -1) != ultimo:
                    ultimo = frame['timestamp']
                    yield _make_camera_frame(frame)
                await asyncio.sleep(_T_CAMERA)
        except asyncio.CancelledError:
            pass

    # -- RPC unario: waypoints -------------------------------------------------

    async def SendWaypoints(self, request, context):
        wps = [(w.x, w.y, w.yaw, w.label) for w in request.waypoints]
        ok  = self._bridge.publish_waypoints(wps)
        return _pb('WaypointResponse',
                   accepted=ok,
                   count=len(wps) if ok else 0,
                   message=f'{len(wps)} waypoints recibidos, goal publicado' if ok else 'Error')


# ── Registro y arranque ───────────────────────────────────────────────────────

def _register(server, bridge):
    from .proto_gen import puzzlebot_pb2_grpc as pb2_grpc
    pb2_grpc.add_PuzzlebotServiceServicer_to_server(_Servicer(bridge), server)


async def _serve(bridge: RosBridge, host: str, port: int):
    server = grpc.aio.server()
    _register(server, bridge)
    server.add_insecure_port(f'{host}:{port}')
    await server.start()
    logger.info('Servidor gRPC escuchando en %s:%d', host, port)

    parar = asyncio.Event()
    loop  = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, parar.set)
    loop.add_signal_handler(signal.SIGINT,  parar.set)

    await parar.wait()
    logger.info('Apagando servidor gRPC…')
    await server.stop(grace=5.0)


def main():
    import argparse
    import threading
    import rclpy

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    parser = argparse.ArgumentParser(description='GUI Bridge — PuzzleBot')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=50051)
    args = parser.parse_args()

    rclpy.init()
    bridge     = RosBridge()
    ros_thread = threading.Thread(target=rclpy.spin, args=(bridge,), daemon=True)
    ros_thread.start()

    try:
        asyncio.run(_serve(bridge, args.host, args.port))
    finally:
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
