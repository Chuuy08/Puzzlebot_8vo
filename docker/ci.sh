#!/bin/bash
set -e

echo '═══════════════════════════════════════'
echo '  PASO 1 — Sintaxis Python'
echo '═══════════════════════════════════════'
ERRORS=0
while IFS= read -r -d '' f; do
  if python3 -m py_compile "$f"; then
    echo "  OK   $f"
  else
    echo "  FAIL $f"
    ERRORS=$((ERRORS+1))
  fi
done < <(find src -name '*.py' ! -path '*/Acts_extras/*' -print0)
if [ $ERRORS -gt 0 ]; then echo "FALLO: $ERRORS archivo(s)"; exit 1; fi
echo 'Sintaxis OK'

echo '═══════════════════════════════════════'
echo '  PASO 2 — Build workspace'
echo '═══════════════════════════════════════'
colcon build \
  --packages-select \
    puzzlebot_challenge \
    puzzlebot_control \
    puzzlebot_description \
    puzzlebot_gazebo \
    puzzlebot_localisation \
  --event-handlers console_direct+

echo '═══════════════════════════════════════'
echo '  PASO 3 — Validar launch files'
echo '═══════════════════════════════════════'
source install/setup.bash
cat > /tmp/check_launch.py << 'PYEOF'
import importlib.util, sys
f = sys.argv[1]
spec = importlib.util.spec_from_file_location('m', f)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
PYEOF
ERRORS=0
while IFS= read -r -d '' f; do
  if python3 /tmp/check_launch.py "$f" 2>&1; then
    echo "  OK   $f"
  else
    echo "  FAIL $f"
    ERRORS=$((ERRORS+1))
  fi
done < <(find src -name '*launch*.py' ! -path '*/Acts_extras/*' -print0)
if [ $ERRORS -gt 0 ]; then echo "FALLO: $ERRORS launch file(s)"; exit 1; fi
echo 'Launch files OK'

echo '═══════════════════════════════════════'
echo '  PASO 4 — Executables disponibles'
echo '═══════════════════════════════════════'
ros2 pkg executables puzzlebot_challenge    || true
ros2 pkg executables puzzlebot_control      || true
ros2 pkg executables puzzlebot_localisation || true

echo ''
echo '✓ CI completo sin errores'
