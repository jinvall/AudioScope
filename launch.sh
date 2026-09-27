#!/usr/bin/env bash
#
# Desktop launcher for Audio Microscope.
#
# Kept deliberately thin: the work is done by the Python entry point, so there
# is one implementation of "launch the system" rather than two that can drift.
# This script only resolves the interpreter and execs it, which also means the
# window is not owned by a shell and closing it really closes the application.
#
# Optional settings, if you want the launcher to behave differently from the
# command line, set these before launching:
#
# Any arguments given here are passed straight through to the entry point, so
# `./launch.sh --source network` does what it says.  Earlier this script dropped
# them, which silently ran the default database instead of the requested one.
#
#   AUDIOMICROSCOPE_SOURCE   network (default, the phone on 8190) or device
#   AUDIOMICROSCOPE_DB       event database (default: events.db)
#   AUDIOMICROSCOPE_NO_CAPTURE=1   open the review window without capturing

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

if [[ -x "./venv/bin/python" ]]; then
    PYTHON="./venv/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

if ! "$PYTHON" -c "import numpy, soundfile" >/dev/null 2>&1; then
    "$PYTHON" - <<'EOF' 2>&1 || true
import sys
print("Audio Microscope: dependencies are missing.", file=sys.stderr)
print("  python3 -m venv venv && ./venv/bin/pip install -r requirements.txt",
      file=sys.stderr)
EOF
    exit 1
fi

ARGS=()
# Default to the phone on 8190, stated here rather than only in Python, so it
# is obvious where the icon's behaviour comes from.
DEFAULT_SOURCE="${AUDIOMICROSCOPE_SOURCE:-network}"
ARGS+=(--source "$DEFAULT_SOURCE")
[[ -n "${AUDIOMICROSCOPE_DB:-}" ]] && ARGS+=(--db "${AUDIOMICROSCOPE_DB}")
[[ "${AUDIOMICROSCOPE_NO_CAPTURE:-0}" == "1" ]] && ARGS+=(--no-capture)
# Anything the user passed wins over the environment defaults.
ARGS+=("$@")

exec "$PYTHON" -m app "${ARGS[@]}"
