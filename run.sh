#!/usr/bin/env bash
#
# Audio Microscope launcher.
#
#   ./run.sh                      launch the system: capture + review window
#   ./run.sh --no-capture         open the review window only
#   ./run.sh --capture-only       run capture without the review window
#   ./run.sh --list-devices       just list input devices
#   ./run.sh --source network     listen for the Android stream on port 8090
#   ./run.sh --seconds 30         capture for 30 seconds and write WAV chunks
#   ./run.sh test input.wav       run the pipeline over a file
#
# Everything runs on CPU.  No GPU, CUDA or vendor runtime is required or used.

set -euo pipefail

cd "$(dirname "$0")"

# Prefer the project virtualenv; fall back to the system interpreter.
if [[ -x "./venv/bin/python" ]]; then
    PYTHON="./venv/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

if ! "$PYTHON" -c "import numpy, soundfile" >/dev/null 2>&1; then
    echo "error: missing dependencies." >&2
    echo "  run: python3 -m venv venv && ./venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

# sounddevice is only needed for live device capture and playback.
if ! "$PYTHON" -c "import sounddevice" >/dev/null 2>&1; then
    echo "warning: sounddevice is not installed; live device capture and" >&2
    echo "         playback are unavailable. File mode still works." >&2
    echo "         install with: ./venv/bin/pip install sounddevice" >&2
    echo >&2
fi

# Default to the documented behavior: show the devices, then capture.
if [[ $# -eq 0 ]]; then
    set -- --list-devices
    "$PYTHON" -m app capture --list-devices
    echo
    echo "Starting live capture. Press Ctrl-C to stop."
    echo
    exec "$PYTHON" -m app capture
fi

# `test` and `analyse` are top-level verbs here.
if [[ "$1" == "test" ]]; then
    shift
    exec "$PYTHON" -m app.test "$@"
fi

if [[ "$1" == "analyse" || "$1" == "analyze" ]]; then
    shift
    exec "$PYTHON" -m app.analyse "$@"
fi

if [[ "$1" == "events" ]]; then
    shift
    exec "$PYTHON" -m app.events "$@"
fi

if [[ "$1" == "gui" || "$1" == "review" ]]; then
    shift
    exec "$PYTHON" -m app.gui "$@"
fi

exec "$PYTHON" -m app.capture "$@"
