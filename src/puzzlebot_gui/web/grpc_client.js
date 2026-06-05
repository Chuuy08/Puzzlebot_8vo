'use strict';

/**
 * PuzzlebotClient
 * Encapsula la comunicación gRPC-Web con el servidor Python.
 * Carga el .proto en tiempo de ejecución (sin compilación).
 *
 * Uso:
 *   const c = new PuzzlebotClient('http://localhost:8443');
 *   await c.init();
 *   c.streamTelemetry(msg => { ... });
 *   await c.sendWaypoints([{x, y, yaw, label}, ...]);
 */
class PuzzlebotClient {
  constructor(proxyUrl) {
    this._url    = proxyUrl;
    this._root   = null;
    this._svc    = null;
  }

  /**
   * Carga el archivo .proto y construye el servicio.
   * Debe llamarse antes de cualquier otro método.
   */
  async init() {
    // El .proto se sirve desde el mismo servidor HTTP que el HTML (puerto 8080)
    this._root = await protobuf.load('puzzlebot.proto');
    this._svc  = this._root.lookupService('puzzlebot.PuzzlebotService');
  }

  // ── Streams de monitoreo ──────────────────────────────────────────────────

  streamTelemetry(onMsg, onErr) {
    this._openStream('StreamTelemetry', {}, onMsg, onErr);
  }

  streamCostmap(onMsg, onErr) {
    this._openStream('StreamCostmap', {}, onMsg, onErr);
  }

  streamPath(onMsg, onErr) {
    this._openStream('StreamPath', {}, onMsg, onErr);
  }

  streamNodeStatus(onMsg, onErr) {
    this._openStream('StreamNodeStatus', {}, onMsg, onErr);
  }

  streamCamera(onMsg, onErr) {
    this._openStream('StreamCamera', {}, onMsg, onErr);
  }

  // ── RPC unario: enviar waypoints ──────────────────────────────────────────

  sendWaypoints(waypoints) {
    return this._unary('SendWaypoints', { waypoints });
  }

  // ── Internals ─────────────────────────────────────────────────────────────

  /**
   * Abre un server-streaming RPC con reconexión automática.
   * - Si el stream termina (done=true) o falla, reintenta tras un delay.
   * - El delay crece con cada fallo consecutivo (backoff), pero se resetea
   *   en cuanto llega el primer mensaje (conexión sana).
   */
  _openStream(method, requestObj, onMsg, onErr) {
    const m       = this._svc.methods[method];
    const ReqType = this._root.lookupType(m.requestType);
    const ResType = this._root.lookupType(m.responseType);

    let failCount = 0;

    const connect = () => {
      const reqBytes = ReqType.encode(ReqType.create(requestObj)).finish();
      const frame    = this._encodeGrpcFrame(reqBytes);

      fetch(`${this._url}/puzzlebot.PuzzlebotService/${method}`, {
        method:  'POST',
        headers: {
          'Content-Type': 'application/grpc-web+proto',
          'X-Grpc-Web':   '1',
        },
        body: frame,
      })
      .then(res => {
        if (!res.ok || !res.body) {
          onErr && onErr(new Error(`HTTP ${res.status}`));
          failCount++;
          setTimeout(connect, Math.min(500 * failCount, 4000));
          return;
        }

        const reader = res.body.getReader();
        let   buf    = new Uint8Array(0);

        const pump = () => reader.read().then(({ done, value }) => {
          if (done) {
            // El servidor cerró el stream limpiamente; reconectar pronto
            setTimeout(connect, 500);
            return;
          }

          // Acumular bytes del chunk recibido
          const next = new Uint8Array(buf.length + value.length);
          next.set(buf);
          next.set(value, buf.length);
          buf = next;

          // Decodificar todos los frames completos disponibles
          while (buf.length >= 5) {
            const compressed = buf[0];
            const msgLen = (buf[1] << 24) | (buf[2] << 16) | (buf[3] << 8) | buf[4];
            if (buf.length < 5 + msgLen) break;

            if (compressed === 0) {
              // Frame de datos (compressed=128 es el trailer gRPC-Web)
              const msgBytes = buf.slice(5, 5 + msgLen);
              try {
                onMsg(ResType.decode(msgBytes).toJSON());
                failCount = 0; // stream sano, resetear backoff
              } catch (e) {
                console.warn(`[grpc] decode error en ${method}:`, e);
              }
            }
            buf = buf.slice(5 + msgLen);
          }

          pump();
        }).catch(err => {
          onErr && onErr(err);
          failCount++;
          setTimeout(connect, Math.min(500 * failCount, 4000));
        });

        pump();
      })
      .catch(err => {
        onErr && onErr(err);
        failCount++;
        setTimeout(connect, Math.min(500 * failCount, 4000));
      });
    };

    connect();
  }

  /**
   * RPC unario: envía un request y espera una sola respuesta.
   */
  async _unary(method, requestObj) {
    const m       = this._svc.methods[method];
    const ReqType = this._root.lookupType(m.requestType);
    const ResType = this._root.lookupType(m.responseType);
    const req     = ReqType.create(requestObj);
    const bytes   = ReqType.encode(req).finish();
    const frame   = this._encodeGrpcFrame(bytes);

    const res = await fetch(`${this._url}/puzzlebot.PuzzlebotService/${method}`, {
      method:  'POST',
      headers: {
        'Content-Type': 'application/grpc-web+proto',
        'X-Grpc-Web':   '1',
      },
      body: frame,
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const raw   = new Uint8Array(await res.arrayBuffer());
    // Saltar el header de 5 bytes del primer frame
    if (raw.length < 5) throw new Error('Respuesta gRPC vacía');
    const msgLen = (raw[1] << 24) | (raw[2] << 16) | (raw[3] << 8) | raw[4];
    const msgBytes = raw.slice(5, 5 + msgLen);
    return ResType.decode(msgBytes).toJSON();
  }

  /**
   * Codifica un mensaje como frame gRPC-Web:
   * [compressed(1B)][length(4B)][data]
   */
  _encodeGrpcFrame(bytes) {
    const frame = new Uint8Array(5 + bytes.length);
    frame[0] = 0; // no comprimido
    frame[1] = (bytes.length >>> 24) & 0xff;
    frame[2] = (bytes.length >>> 16) & 0xff;
    frame[3] = (bytes.length >>>  8) & 0xff;
    frame[4] = (bytes.length >>>  0) & 0xff;
    frame.set(bytes, 5);
    return frame;
  }

}
