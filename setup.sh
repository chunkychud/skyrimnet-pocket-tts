#!/bin/bash
# ============================================================
# setup.sh - Linux setup script for the server
# 1) Verify Python 3.10-3.14 is installed and works
# 2) Verify weights/tts_skyrimnet.safetensors exists
# 3) Create .venv and pip install requirements
# 4) Copy weights/skyrimnet.yaml into .venv/lib/pythonX.Y/site-packages/pocket_tts/config/
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

  # Resolve script directory robustly (works when sourced or executed)
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  # --- Step 2: Verify weights file exists early ---
  WEIGHTS_FILE="$SCRIPT_DIR/weights/tts_skyrimnet.safetensors"
  if [ ! -f "$WEIGHTS_FILE" ]; then
    die "Missing required weights file: $WEIGHTS_FILE"
  fi
  echo "[OK] Found weights file: $WEIGHTS_FILE"

  # --- Step 1: Find a working Python in allowed versions ---
  PY_CMD=""
  PY_VER=""

  for CAND in python3 python; do
    if command -v "$CAND" >/dev/null 2>&1; then
      for V in 3.14 3.13 3.12 3.11 3.10; do
        if "$CAND" -c "import sys; assert sys.version_info[:2]==tuple(map(int,'$V'.split('.')))" 2>/dev/null; then
          PY_CMD="$CAND"
          PY_VER="$V"
          break 2
        fi
      done
    fi
  done

  if [ -z "$PY_CMD" ]; then
    die "Could not find a working Python 3.10-3.14. Install Python 3.10-3.14 and re-run."
  fi

  # Optional: check for Ubuntu-specific venv package (informational only)
  if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [ "$ID" = "ubuntu" ]; then
      if dpkg -s "python${PY_VER%%.*}-venv" &>/dev/null; then
        echo "Running on Ubuntu and python${PY_VER%%.*}-venv is installed."
      else
        echo "Running on Ubuntu but python${PY_VER%%.*}-venv may not be installed."
      fi
    else
      echo "Not running on Ubuntu."
    fi
  else
    echo "Could not determine the operating system."
  fi

  echo "[OK] Using Python $PY_VER via: $PY_CMD"

  # --- Step 3: Create venv + install requirements ---
  VENV_DIR="$SCRIPT_DIR/.venv"
  if [ ! -d "$VENV_DIR/bin" ]; then
    echo
    echo "[INFO] Creating virtual environment in '.venv' ..."
    "$PY_CMD" -m venv "$VENV_DIR" || die "Failed to create venv in $VENV_DIR"
  else
    echo
    echo "[OK] Virtual environment already exists: $VENV_DIR"
  fi

  VENV_PY="$VENV_DIR/bin/python"

  echo
  echo "[INFO] Upgrading pip/setuptools/wheel ..."
  "$VENV_PY" -m pip install --upgrade pip setuptools wheel || die "Failed to upgrade pip tooling"

  if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
    die "requirements.txt not found in repo root: $SCRIPT_DIR/requirements.txt"
  fi

  echo
  echo "[INFO] Installing requirements ..."
  "$VENV_PY" -m pip install -r "$SCRIPT_DIR/requirements.txt" || die "pip install -r requirements.txt failed"

  # --- Step 4: Copy weights to pocket_tts config directory inside venv ---
  # Use the venv python to compute the actual site-packages path
  SITE_PKG_DIR="$("$VENV_PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")"
  DST_DIR="$SITE_PKG_DIR/pocket_tts/config"

  if [ ! -d "$DST_DIR" ]; then
    die "Destination folder not found: $DST_DIR. pocket_tts may not have installed correctly."
  fi

  echo
  echo "[INFO] Copying weights/skyrimnet.yaml -> $DST_DIR"
  if command -v rsync &>/dev/null; then
    rsync -a "$SCRIPT_DIR/weights/skyrimnet.yaml" "$DST_DIR/" || die "rsync failed copying skyrimnet.yaml to pocket_tts config"
  else
    cp -r "$SCRIPT_DIR/weights/skyrimnet.yaml" "$DST_DIR/" || die "cp failed copying weights to pocket_tts config"
  fi
  
  echo
  echo "[OK] Setup complete."
  echo "    Next: use runserver.sh to start the API."
  echo
}

main "$@"