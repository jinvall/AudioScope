#!/usr/bin/env python3
"""Render the review window to a PNG.

Useful for checking the interface without a screenshot tool: the window is
constructed exactly as ``python -m app.gui`` constructs it, driven the way a
reviewer would drive it, then grabbed with ``QWidget.grab()`` - Qt's own
render, so what is written is what is on screen.

    python tools/gui_snapshot.py --db events.db --out shot.png
    QT_QPA_PLATFORM=offscreen python tools/gui_snapshot.py --out shot.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Run from anywhere: make the project importable, as run.sh does.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=os.environ.get(
        "AUDIOMICROSCOPE_DB", "events.db"))
    parser.add_argument("--theme", default="dark", choices=("dark", "light"))
    parser.add_argument("--out", default="gui-snapshot.png")
    parser.add_argument(
        "--tab", default="Event",
        help="which inspector tab to show: Event, Acoustics, Source, Review, "
             "Details, Spectrogram, Similar, Separate",
    )
    parser.add_argument(
        "--events", type=int, default=0,
        help="load audio and a waveform for the Nth event (0-based). Default "
             "0, the newest.",
    )
    parser.add_argument("--no-audio", action="store_true",
                        help="skip loading audio, to show the empty state")
    args = parser.parse_args(argv)

    from PyQt5.QtWidgets import QApplication

    from app.config import AppConfig
    from app.gui.controller import open_controller
    from app.gui.main_window import MainWindow
    from app.gui.theme import build_stylesheet, load_theme

    application = QApplication.instance() or QApplication(sys.argv[:1])
    application.setApplicationName("Audio Microscope")

    config = AppConfig().validate()
    theme = load_theme(args.theme)
    application.setStyleSheet(build_stylesheet(theme))

    controller = open_controller(args.db, config=config)
    window = MainWindow(controller, theme, db_path=args.db)
    window.resize(1280, 820)
    window.show()
    application.processEvents()

    rows = controller.list_events()
    if not rows:
        print("no events to show", file=sys.stderr)
        return 2

    target = rows[min(args.events, len(rows) - 1)]
    window.select_event(target.event_id)
    if not args.no_audio:
        # Pump the background work so the snapshot shows a real waveform.
        for _ in range(200):
            window._pump_pending()
            application.processEvents()
            if controller.cached_envelope(target.event_id) is not None:
                break
            time.sleep(0.02)
        window._pump_pending()
        application.processEvents()

    for index in range(window.tabs.count()):
        if window.tabs.tabText(index) == args.tab:
            window.tabs.setCurrentIndex(index)
            break
    for _ in range(120):
        application.processEvents()
        if args.tab != "Spectrogram" or window.spectrogram._image is not None:
            break
        window._pump_spectrogram()
        time.sleep(0.02)
    application.processEvents()

    pixmap = window.grab()
    pixmap.save(args.out, "PNG")
    print(f"wrote {os.path.abspath(args.out)} "
          f"({pixmap.width()}x{pixmap.height()}) "
          f"event={target.event_id} tab={args.tab}")
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
