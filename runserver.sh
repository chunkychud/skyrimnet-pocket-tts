#!/bin/bash
# ...existing code...

set -euo pipefail

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

# If someone accidentally sources this script, stop to avoid changing their shell state.
if [ "${BASH_SOURCE[0]}" != "$0" ]; then
  echo "[ERROR] Do not source this script. Run it: ./runserver.sh" >&2
  return 1 2>/dev/null || exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || die "Failed to cd to $SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
PY="$VENV_DIR/bin/python"

if [ ! -x "$PY" ]; then
  die "Virtual environment not found or python not executable. Run setup.sh first. Expected: $PY"
fi

echo
echo "=== Starting SkyrimNet TTS API ==="
echo
echo "[INFO] Using Python: $PY"
echo

# Do NOT source activate in this launcher (can cause nested shell state). If you want to
# activate interactively, run: source .venv/bin/activate
# source "$VENV_DIR/bin/activate" || die "Failed to activate venv"

ENTRYPOINT="$SCRIPT_DIR/src/skyrimnet_api.py"
if [ ! -f "$ENTRYPOINT" ]; then
  die "Could not find entrypoint $ENTRYPOINT. Update ENTRYPOINT in runserver.sh"
fi

echo "[INFO] Running: $ENTRYPOINT"
echo

# Replace this shell with the python process to avoid extra shell levels
exec "$PY" "$ENTRYPOINT"
# ...existing code...