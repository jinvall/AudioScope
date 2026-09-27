"""Single entry point: bring the whole system up.

``python -m app`` with no arguments does the thing a person actually wants when
they start the application: it opens the review workstation and, alongside it,
starts capture feeding the same database, so events appear in the list as they
are detected.

    microphone / phone -> capture -> detection -> events.db -> review GUI

Capture runs as a **separate process**, not inside the GUI.  That is deliberate
and it is a phase boundary, not an implementation detail: Phase 5 specified that
the GUI is a client of the backend, must not start or reconfigure detection,
and must perform no analysis.  Running capture as a child process keeps that
separation literal - the GUI only ever reads the database the capture process
writes, and the two can be started, stopped and restarted independently.

The GUI already tolerates this: it notices new events through the database's
change marker, so it does not need to own or watch the capture process at all.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import Optional

DEFAULT_DB = os.environ.get("AUDIOMICROSCOPE_DB", "events.db")
DEFAULT_EVENTS = "events"
DEFAULT_RECORDINGS = "recordings"


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def python_executable() -> str:
    """The interpreter to use for child processes.

    ``sys.executable`` so a child runs in the same virtualenv, falling back to
    the project's ``venv`` when the launcher was started some other way.
    """
    if sys.executable and "python" in os.path.basename(sys.executable):
        return sys.executable
    candidate = os.path.join(project_root(), "venv", "bin", "python")
    return candidate if os.path.exists(candidate) else sys.executable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description=(
            "Launch Audio Microscope: capture feeds the review workstation."
        ),
        epilog=(
            "Sub-commands are also available for individual stages: capture, "
            "test, analyse, events, gui, separate."
        ),
    )
    parser.add_argument(
        "--source", choices=("network", "device"), default="network",
        help=(
            "where audio comes from. Defaults to network, because this "
            "project's source is the phone streaming to port 8190; the earlier "
            "default of device silently recorded the local microphone while the "
            "phone found nothing listening. Use --source device for a local "
            "input."
        ),
    )
    parser.add_argument("--device", default=None,
                        help="input device index or name for the device source")
    parser.add_argument(
        "--port", type=int, default=8190,
        help="network stream port (default: 8190)",
    )
    parser.add_argument("--db", default=DEFAULT_DB,
                        help=f"event database (default: {DEFAULT_DB})")
    parser.add_argument("--events", default=DEFAULT_EVENTS,
                        help="event directory root (default: events)")
    parser.add_argument("--recordings", default=DEFAULT_RECORDINGS,
                        help="continuous recording directory (default: recordings)")
    parser.add_argument("--theme", choices=("dark", "light"), default="dark",
                        help="interface theme (default: dark)")
    parser.add_argument(
        "--seconds", type=float, default=None,
        help="stop capture after this long (default: until the window closes)",
    )
    parser.add_argument(
        "--no-capture", action="store_true",
        help="open the review window only, with no capture",
    )
    parser.add_argument(
        "--no-splash", action="store_true",
        help="skip the start-the-stream splash and open the window directly",
    )
    parser.add_argument(
        "--capture-only", action="store_true",
        help="run capture without opening the review window",
    )
    return parser


# ----------------------------------------------------------------------
class CaptureProcess:
    """Capture in a child process, so the GUI stays a client of the backend."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.process: Optional[subprocess.Popen] = None
        self.log_path: Optional[str] = None

    def command(self) -> list:
        cmd = [
            python_executable(), "-u", "-m", "app.capture",
            "--source", self.args.source,
            "--db", self.args.db,
            "--events", self.args.events,
            "--output", self.args.recordings,
        ]
        if self.args.source == "network":
            cmd += ["--port", str(self.args.port)]
        if self.args.device:
            cmd += ["--device", str(self.args.device)]
        if self.args.seconds:
            cmd += ["--seconds", str(self.args.seconds)]
        return cmd

    def start(self) -> bool:
        if self.process is not None:
            return True
        os.makedirs(self.args.recordings, exist_ok=True)
        # Capture's own output would fight the GUI's for the terminal, and it is
        # long-running, so it goes to a log the user can read afterwards.
        self.log_path = os.path.join(
            self.args.recordings, "capture.log"
        )
        try:
            self.log = open(self.log_path, "w", encoding="utf-8")
        except OSError:
            self.log = None
        try:
            self.process = subprocess.Popen(
                self.command(),
                cwd=project_root(),
                stdout=self.log or subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            print(f"warning: could not start capture: {exc}", file=sys.stderr)
            self.process = None
            return False
        return True

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self, timeout: float = 8.0) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            # Ask nicely first, so the recorder closes its final chunk.
            try:
                self.process.send_signal(signal.SIGINT)
            except OSError:
                pass
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline and self.process.poll() is None:
                time.sleep(0.1)
            if self.process.poll() is None:
                try:
                    self.process.terminate()
                except OSError:
                    pass
        if getattr(self, "log", None) is not None:
            try:
                self.log.close()
            except OSError:
                pass
        self.process = None

    def report(self) -> None:
        if self.log_path and os.path.exists(self.log_path):
            size = os.path.getsize(self.log_path)
            print(f"  capture log: {os.path.abspath(self.log_path)} ({size} bytes)")


# ----------------------------------------------------------------------
def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv or [])

    if args.capture_only:
        return _run_capture_in_foreground(args)

    if not args.no_capture:
        capture = CaptureProcess(args)
        if capture.start():
            _say("capture started")
            # Give the port reservation and device open a moment, so the review
            # window opens onto a database that is already being written.
            time.sleep(1.5)
        else:
            capture = None
    else:
        capture = None
        _say("capture not started (--no-capture)")

    if not args.no_capture:
        # Create the schema now.  The review window refuses to open a missing
        # database - correctly, since standalone there would be nothing to show
        # and never anything to write - but when capture is running the window
        # should open onto an empty list and fill as events arrive.  Opening and
        # closing the store here creates the file, and SQLite handles the
        # concurrent access the capture child is already making.
        _ensure_database(args.db)

    _say("opening review window; close it to stop capture")
    return _open_gui(args, capture)


def _ensure_database(db_path: str) -> bool:
    """Create the event database if it does not exist.  Returns True if ready."""
    if os.path.exists(db_path):
        return True
    try:
        from .events.database import EventDatabase

        EventDatabase(db_path).close()
    except Exception as exc:
        print(
            f"warning: could not create {db_path}: {exc}", file=sys.stderr
        )
        return False
    return True


def _say(message: str) -> None:
    """Print a status line that appears immediately.

    Launched from a desktop icon there is no terminal, so stdout is block
    buffered and every status line would otherwise appear at once, on exit, if
    the process was killed rather than closed.
    """
    print(message, flush=True)


def _open_gui(args: argparse.Namespace, capture) -> int:
    try:
        from .gui.app import run_gui
    except ImportError as exc:
        print(
            f"error: the review window needs PyQt5, which could not be "
            f"imported: {exc}\n"
            "       capture still works: python -m app.capture",
            file=sys.stderr,
        )
        if capture is not None:
            capture.stop()
        return 2
    try:
        return run_gui(
            args.db, theme=args.theme,
            show_splash=not args.no_splash,
            source=args.source, port=args.port,
            # The same tree capture was told to write, so separations land
            # beside the events they came from.
            events_root=args.events,
            # So the window's Live tab reads the status this capture process
            # is publishing, not a default it happens to share.
            capture_dir=args.recordings,
            capture_log=(capture.log_path if capture is not None else None),
        )
    finally:
        if capture is not None:
            _say("stopping capture")
            capture.stop()
            capture.report()


def _run_capture_in_foreground(args: argparse.Namespace) -> int:
    cmd = [
        python_executable(), "-u", "-m", "app.capture",
        "--source", args.source,
        "--db", args.db,
        "--events", args.events,
        "--output", args.recordings,
    ]
    if args.source == "network":
        cmd += ["--port", str(args.port)]
    if args.device:
        cmd += ["--device", str(args.device)]
    if args.seconds:
        cmd += ["--seconds", str(args.seconds)]
    return subprocess.call(cmd, cwd=project_root())


if __name__ == "__main__":
    raise SystemExit(main())
