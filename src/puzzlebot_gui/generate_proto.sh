#!/bin/bash
# Genera los stubs Python desde puzzlebot.proto.
# Ejecutar desde la raíz del paquete: ./generate_proto.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 -m grpc_tools.protoc \
  -I"${SCRIPT_DIR}/proto" \
  --python_out="${SCRIPT_DIR}/puzzlebot_gui/proto_gen" \
  --grpc_python_out="${SCRIPT_DIR}/puzzlebot_gui/proto_gen" \
  "${SCRIPT_DIR}/proto/puzzlebot.proto"

# Corregir imports relativos generados por protoc (usan imports absolutos por defecto)
sed -i 's/^import puzzlebot_pb2/from . import puzzlebot_pb2/' \
  "${SCRIPT_DIR}/puzzlebot_gui/proto_gen/puzzlebot_pb2_grpc.py"

echo "Stubs generados en puzzlebot_gui/proto_gen/"
