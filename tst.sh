#!/bin/bash

# ============================================================
# setup.sh - Linux setup script for the server
# 1) Verify Python 3.10-3.14 is installed and works
# 2) Verify weights/tts_skyrimnet.safetensors exists
# 3) Create .venv and pip install requirements
# 4) Copy weights/* into .venv/lib/pythonX.Y/site-packages/pocket_tts/config/
#
# On any error, the script will exit with a non-zero status.
# ============================================================

# --- Helper: print error, exit ---
die() {
  echo "[ERROR] $*" >&2
  exit 1
}

main() {
  echo
  echo "=== SkyrimNet TTS Server Setup ==="
  echo

  # --- Step 2: Verify weights file exists early ---
  WEIGHTS_FILE="$(dirname "$0")/weights/tts_skyrimnet.safetensors"
  if [ ! -f "$WEIGHTS_FILE" ]; then
    die "Missing required weights file: $WEIGHTS_FILE"
  fi
  echo "[OK] Found weights file: $WEIGHTS_FILE"

  # --- Step 1: Find a working Python in allowed versions ---
  PY_CMD=""
  PY_VER=""

  # Try python3 first
  if command -v python3 &>/dev/null; then
    for V in 3.14 3.13 3.12 3.11 3.10; do
      if python3 -c "import sys; assert sys.version_info[:2]==tuple(map(int,'$V'.split('.')))" 2>/dev/null; then
        PY_CMD="python3"
        PY_VER="$V"
        break
      fi
    done
  fi

  if [ -z "$PY_CMD" ]; then
    die "Could not find a working Python 3.10-3.14. Install Python 3.10-3.14 and re-run."
  fi

  echo "[OK] Using Python $PY_VER via: $PY_CMD"

  # --- Step 3: Create venv + install requirements ---
  VENV_DIR="$(dirname "$0")/.venv"
  if [ ! -d "$VENV_DIR/bin" ]; then
    echo
    echo "[INFO] Creating virtual environment in '.venv' ..."
    $PY_CMD -m venv "$VENV_DIR" || die "Failed to create venv in $VENV_DIR"
  else
    echo
    echo "[OK] Virtual environment already exists: $VENV_DIR"
  fi

  echo
  echo "[INFO] Upgrading pip/setuptools/wheel ..."
  "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel || die "Failed to upgrade pip tooling"

  if [ ! -f "$(dirname "$0")/requirements.txt" ]; then
    die "requirements.txt not found in repo root: $(dirname "$0")/requirements.txt"
  fi

  echo
  echo "[INFO] Installing requirements ..."
  "$VENV_DIR/bin/python" -m pip install -r "$(dirname "$0")/requirements.txt" || die "pip install -r requirements.txt failed"

  # --- Step 4: Copy weights to pocket_tts config directory inside venv ---
  DST_DIR="$VENV_DIR/lib/python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/pocket_tts/config"
  if [ ! -d "$DST_DIR" ]; then
    die "Destination folder not found: $DST_DIR. pocket_tts may not have installed correctly."
  fi

  echo
  echo "[INFO] Copying weights/* -> $DST_DIR"
  if command -v rsync &>/dev/null; then
    rsync -a "$(dirname "$0")/weights/" "$DST_DIR/" || die "rsync failed copying weights to pocket_tts config"
  else
    cp -r "$(dirname "$0")/weights/"* "$DST_DIR/" || die "cp failed copying weights to pocket_tts config"
  fi

  echo
  echo "[OK] Setup complete."
  echo "    Next: use run_server.sh to start the API."
  echo
}

main "$@"