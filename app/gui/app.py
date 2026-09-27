"""GUI entry point: argument handling, Qt setup, and the event loop.

Kept separate from the window so the window can be constructed in tests without
argument parsing, and so a headless smoke check can build the whole thing
without an event loop.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from .theme import THEME_PACK_DIR, Theme, build_stylesheet, load_theme


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.gui",
        description="Audio Microscope event review workstation.",
    )
    parser.add_argument(
        "--db", default=os.environ.get("AUDIOMICROSCOPE_DB", "events.db"),
        help="event database to review (default: events.db)",
    )
    parser.add_argument(
        "--recordings", default=os.environ.get(
            "AUDIOMICROSCOPE_RECORDINGS", "recordings"),
        help="directory capture publishes its live status into (default: "
             "recordings). The Live tab reads it from here.",
    )
    parser.add_argument(
        "--events", default=os.environ.get("AUDIOMICROSCOPE_EVENTS", "events"),
        help="event directory tree the capture wrote (default: events). "
             "Separation results are written beside the event they came from, "
             "so this has to be the same tree capture used.",
    )
    parser.add_argument(
        "--theme", choices=("dark", "light"), default="dark",
        help="theme flavour from the design system (default: dark)",
    )
    parser.add_argument(
        "--theme-dir", default=None,
        help="override the theme pack directory (for testing)",
    )
    parser.add_argument(
        "--no-splash", action="store_true",
        help="skip the start-the-stream splash and go straight to the window",
    )
    parser.add_argument(
        "--port", type=int, default=8190,
        help="port shown on the splash (default: 8190)",
    )
    parser.add_argument(
        "--recordings", default="recordings",
        help="continuous recording directory, where the capture log lives",
    )
    return parser


def run_gui(db_path: str, theme: str = "dark", app=None,
            show_splash: bool = True, source: str = "network",
            port: int = 8190, capture_log: Optional[str] = None,
            events_root: Optional[str] = None,
            capture_dir: Optional[str] = None) -> int:
    """Open the splash, then the review window on an existing database.

    Split out from :func:`main` so the single entry point
    (:mod:`app.launch`) can hand over a database it has already started
    writing to, without going through argv.

    The splash is not decoration.  The Android sender stops retrying after a
    fixed number of attempts, so whoever starts the system must also start the
    stream; the splash makes that an explicit instruction rather than a silent
    race.  It is dismissed by any key, and reports the real receive counters
    while it waits.
    """
    from PyQt5.QtWidgets import QApplication

    from ..config import AppConfig
    from .controller import open_controller
    from .main_window import MainWindow

    if not os.path.exists(db_path):
        # With capture running the launcher creates the schema up front, so this
        # only fires when nothing is going to write here.
        print(
            f"No event database at {os.path.abspath(db_path)}.",
            file=sys.stderr,
        )
        return 2

    config = AppConfig().validate()
    resolved = load_theme(theme)
    if not resolved.available:
        print(
            f"warning: theme pack not found at {THEME_PACK_DIR}; "
            "falling back to default styling",
            file=sys.stderr,
        )

    application = app or QApplication.instance() or QApplication(sys.argv[:1])
    application.setApplicationName("Audio Microscope")
    application.setStyleSheet(build_stylesheet(resolved))

    try:
        controller = open_controller(
            db_path, config=config, events_root=events_root,
            capture_dir=capture_dir,
        )
    except Exception as exc:
        print(f"error: cannot open {db_path}: {exc}", file=sys.stderr)
        return 2

    if show_splash:
        _run_splash(application, resolved, db_path, source, port, controller,
                    capture_log=capture_log)

    window = MainWindow(controller, resolved, db_path=db_path)
    window.show()
    return application.exec_()


def _run_splash(application, theme, db_path: str, source: str, port: int,
                 controller=None, capture_log: Optional[str] = None) -> None:
    """Show the splash and block until the operator confirms.

    While it waits, a short timer reads the real receive counter so the screen
    can say whether audio is actually arriving.  Capture runs in another
    process, so its log is where that counter lives; nothing here estimates it.
    """
    from PyQt5.QtCore import QEventLoop, QTimer

    from .splash import SplashWindow

    splash = SplashWindow(theme, db_path=db_path, source=source, port=port,
                         capture_log=capture_log)
    splash.show()

    def refresh():
        received, connected = splash.poll_capture_log()
        splash.set_stream_state(received, connected)

    refresh()
    timer = QTimer(splash)
    timer.setInterval(400)
    timer.timeout.connect(refresh)
    timer.start()
    # SplashWindow is a QWidget, and a QWidget has no exec_(): that method
    # belongs to QDialog and QApplication.  Calling it raised
    # AttributeError on every default launch, killing the process before the
    # review window was built - and with Terminal=false in the .desktop file
    # nobody saw the traceback, so the icon looked inert.
    #
    # To block until the operator confirms, run a local event loop and quit it
    # from the splash's own dismissed signal, which exists for exactly this.
    loop = QEventLoop()
    splash.dismissed.connect(loop.quit)
    try:
        loop.exec_()
    finally:
        timer.stop()
        splash.close()


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv or sys.argv[1:])

    from ..config import AppConfig

    config = AppConfig().validate()
    if not os.path.exists(args.db):
        print(
            f"No event database at {os.path.abspath(args.db)}.\n"
            "Run a capture first, for example:\n"
            "  python -m app.capture --seconds 60 --db " + args.db,
            file=sys.stderr,
        )
        return 2

    theme = load_theme(args.theme, args.theme_dir)
    if not theme.available:
        print(
            f"warning: theme pack not found at {THEME_PACK_DIR}; "
            "falling back to default styling",
            file=sys.stderr,
        )

    from PyQt5.QtWidgets import QApplication

    application = QApplication.instance() or QApplication(sys.argv[:1])
    application.setApplicationName("Audio Microscope")
    application.setStyleSheet(build_stylesheet(theme))

    from .controller import open_controller
    from .main_window import MainWindow

    try:
        controller = open_controller(
            args.db, config=config, events_root=args.events,
            capture_dir=args.recordings,
        )
    except Exception as exc:
        print(f"error: cannot open {args.db}: {exc}", file=sys.stderr)
        return 2

    window = MainWindow(controller, theme, db_path=args.db)
    window.show()
    return application.exec_()


def launch(argv: Optional[list] = None, offscreen: bool = False):
    """Build the window without entering the event loop.

    Used by tests and by a headless smoke check: constructs the controller,
    theme and window exactly as ``main`` does, and returns them so a caller can
    drive the widgets directly.  Nothing is painted and no audio is started.
    """
    if offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    application = QApplication.instance() or QApplication(sys.argv[:1])

    from ..config import AppConfig

    config = AppConfig().validate()
    theme = load_theme("dark")
    application.setStyleSheet(build_stylesheet(theme))

    from .controller import open_controller
    from .main_window import MainWindow

    controller = open_controller("events.db", config=config)
    window = MainWindow(controller, theme, db_path="events.db")
    return application, window, controller
