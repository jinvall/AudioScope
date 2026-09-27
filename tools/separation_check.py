"""End-to-end proof of the Phase 4 acceptance chain, with the real model.

    ./venv/bin/python tools/separation_check.py

tasks/PHASE_04_SEPARATION.md ends with an acceptance test, and AGENTS.md
section 26 lists the chain that has to work.  This proves it end to end:
select an event, type a query, let the real model run, then play and save what
it produced - through the same controller, worker, panel, transport and save
handler the window uses.

The unit tests in tests/test_separation.py cover the application layer
against a fake model, because every bug found while building this was in that
layer.  This is the counterpart: it cannot be faked, and it is slow - minutes
of CPU, not seconds - so it is a tool rather than part of the test suite.

Exits non-zero if any step of the chain fails.

AGENTS.md section 26 requires this sequence to work:

    event captured -> user selects it -> query entered -> separation runs ->
    isolated WAV created -> isolated WAV plays -> isolated WAV can be saved

repeated for several queries and for a manually selected region.

Everything here goes through the same path the GUI uses: the controller's
separation worker, the real AudioSep model process, the SeparationPanel's
attempt list, the transport, and the save handler.  Nothing is stubbed except
the file dialog, which is modal and cannot be driven headless - it is
replaced with a fixed destination so the code under test still runs.

"""

import os
import sys
import tempfile

sys.path.insert(0, "/home/jason/audioscope")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtCore import QElapsedTimer, QThread
from PyQt5.QtWidgets import QApplication, QFileDialog

from app.config import AppConfig
from app.events.database import EventDatabase
from app.events.event import Event
from app.events.store import EventStore
from app.gui.controller import ReviewController
from app.gui.main_window import MainWindow
from app.gui.theme import load_theme

ROOT = os.environ.get("SEPARATION_CHECK_ROOT",
                      os.path.join(tempfile.gettempdir(), "separation_check"))
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def spin(app, ms):
    """Run the event loop, so queued signals and the pump timer actually fire."""
    timer = QElapsedTimer()
    timer.start()
    while timer.elapsed() < ms:
        app.processEvents()
        QThread.msleep(10)


def wait_for_separation(app, controller, window, attempts_wanted, timeout_ms=600_000):
    """Wait until the worker is idle and the panel has the expected attempts."""
    timer = QElapsedTimer()
    timer.start()
    last = ""
    while timer.elapsed() < timeout_ms:
        spin(app, 100)
        status = controller.separation_status()
        worker = status.get("worker") or {}
        attempts = window.separation.attempts_list.count()
        line = (f"busy={worker.get('busy')} pending={worker.get('pending')} "
                f"attempts={attempts}")
        if line != last:
            print(f"    ... {line}")
            last = line
        if not worker.get("busy") and not worker.get("pending") \
                and attempts >= attempts_wanted:
            return True
    return False


def correlation(a_path, b_path):
    import soundfile as sf

    a, _ = sf.read(a_path, dtype="float32")
    b, _ = sf.read(b_path, dtype="float32")
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b) + 1e-20))


def main():
    import shutil

    shutil.rmtree(ROOT, ignore_errors=True)
    os.makedirs(ROOT)
    events_root = os.path.join(ROOT, "events")
    db_path = os.path.join(ROOT, "events.db")

    # ---- a real event with real audio on disk ------------------------
    rate = 48_000
    rng = np.random.default_rng(3)
    audio = (0.0006 * rng.standard_normal(rate * 2)).astype(np.float32)  # room
    t = np.arange(int(rate * 0.4)) / rate
    audio[:t.size] += (0.4 * np.sin(2 * np.pi * 85 * t) * np.exp(-t * 6)).astype(
        np.float32
    )                                                            # a thump burst
    store = EventStore(events_root)
    event = Event(event_id="event_000001", start_seconds=0.0, end_seconds=2.0,
                  sample_rate=rate, classification="probe")
    store.save(event, audio)

    db = EventDatabase(db_path)
    db.store_event(
        event_id="event_000001", timestamp="2026-09-27T14:00:00", duration=2.0,
        metadata={"classification": {"label": "probe"}},
        audio_path=event.audio_path, classification="probe",
    )
    size = os.path.getsize(event.audio_path)
    db.set_audio_state("event_000001", audio_bytes=size, available=True)
    db.set_audio_path("event_000001", event.audio_path)

    config = AppConfig().validate()
    controller = ReviewController(db, sample_rate=rate, config=config,
                               events_root=events_root)

    if not controller.separation_available():
        print("source separation is not available; cannot prove the chain")
        return 2

    app = QApplication([])
    window = MainWindow(controller, load_theme())
    window.resize(1280, 900)

    # ---- select the event, as a user would ---------------------------
    titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    window.select_event("event_000001")
    spin(app, 1200)
    window.tabs.setCurrentIndex(titles.index("Separate"))
    spin(app, 600)
    check("event selected and audio loaded", window._event_duration == 2.0,
          f"duration {window._event_duration}")
    check("Separate tab reachable", "Separate" in titles, str(titles))

    # ---- query 1: whole event ---------------------------------------
    print("\n-- whole-event separation --")
    window.separation.query_box.setCurrentText("footsteps")
    window.separation.separate_requested.emit("footsteps", False)
    check("job accepted", wait_for_separation(app, controller, window, 1),
          f"last error: {controller.separation_status()['worker'].get('last_error')}")

    attempts = controller.separation_attempts(window._stored)
    check("one attempt recorded", len(attempts) == 1, str([a.name for a in attempts]))
    attempt = attempts[0] if attempts else None
    if attempt is None:
        return 1
    check("attempt succeeded", attempt.ok, str(attempt.error))
    check("attempt row visible in the panel",
          window.separation.attempts_list.count() == 1)
    check("isolated WAV exists on disk",
          bool(attempt.isolated_path) and os.path.exists(attempt.isolated_path),
          str(attempt.isolated_path))
    check("metadata records the query and the cost",
          attempt.query == "footsteps" and attempt.realtime_ratio is not None,
          f"query={attempt.query!r} ratio={attempt.realtime_ratio}")
    check("cost is reported to the user", "realtime" in attempt.summary(),
          attempt.summary())

    # The acceptance test: the output must be a real separation, not a copy
    # and not a filter of the input.
    corr = correlation(event.audio_path, attempt.isolated_path)
    check("output is a separation, not a copy of the input", corr < 0.95,
          f"correlation with the mixture {corr:.3f}")

    # ---- the isolated audio plays -----------------------------------
    print("\n-- playback of the isolated audio --")
    window._on_compare_separation(attempt.isolated_path)
    spin(app, 2500)
    check("isolated audio loaded into the transport",
          window.player.is_loaded and window._playback_source == "isolated",
          f"loaded={window.player.is_loaded} source={window._playback_source}")
    check("transport enabled for the isolated audio",
          window.play_button.isEnabled())
    window.toggle_play()
    spin(app, 500)

    check("playback state is playing or paused",
          str(window.player.state) in ("PlaybackState.PLAYING",
                                       "PlaybackState.PAUSED"),
          str(window.player.state))
    window.player.stop()

    # ---- save --------------------------------------------------------
    print("\n-- saving the isolated audio --")
    destination = os.path.join(ROOT, "saved_isolated.wav")
    original_dialog = QFileDialog.getSaveFileName
    QFileDialog.getSaveFileName = staticmethod(
        lambda *a, **k: (destination, "")
    )
    try:
        window._on_save_separation(attempt.isolated_path)
    finally:
        QFileDialog.getSaveFileName = original_dialog
    check("save handler wrote the file",
          os.path.exists(destination) and os.path.getsize(destination) > 0,
          f"{os.path.getsize(destination) if os.path.exists(destination) else 0} bytes")
    check("saved file is the isolated audio, byte for byte",
          os.path.exists(destination)
          and open(destination, "rb").read()
          == open(attempt.isolated_path, "rb").read())
    check("the original is untouched by all of this",
          os.path.exists(event.audio_path)
          and os.path.getsize(event.audio_path) == size)

    # ---- query 2: a different query must not overwrite ---------------
    print("\n-- second query must not overwrite the first --")
    window.separation.separate_requested.emit("a person moving", False)
    check("second job completed", wait_for_separation(app, controller, window, 2),
          f"last error: {controller.separation_status()['worker'].get('last_error')}")
    attempts = controller.separation_attempts(window._stored)
    check("two independent attempts", len(attempts) == 2,
          str([a.name for a in attempts]))
    if len(attempts) == 2:
        check("both isolated files still exist",
              all(a.isolated_path and os.path.exists(a.isolated_path)
                  for a in attempts))
        check("the two queries produced different audio",
              correlation(attempts[0].isolated_path,
                          attempts[1].isolated_path) < 0.999)
        check("first attempt's query preserved",
              attempts[0].query == "footsteps"
              and attempts[1].query == "a person moving",
              f"{attempts[0].query!r} / {attempts[1].query!r}")

    # ---- a manually selected region ----------------------------------
    print("\n-- manually selected region --")
    window._on_region_changed(0.25, 0.75)          # 0.5 s .. 1.5 s
    spin(app, 300)
    check("region shown in the panel",
          "0.50s to 1.50s" in window.separation.region_label.text(),
          window.separation.region_label.text())
    window.separation.separate_requested.emit("scraping", True)
    check("region job completed", wait_for_separation(app, controller, window, 3),
          f"last error: {controller.separation_status()['worker'].get('last_error')}")
    attempts = controller.separation_attempts(window._stored)
    check("three attempts now", len(attempts) == 3,
          str([a.name for a in attempts]))
    if len(attempts) == 3:
        region_attempt = attempts[2]
        check("region attempt succeeded", region_attempt.ok, str(region_attempt.error))
        check("the exact separated region was kept for review",
              os.path.exists(os.path.join(region_attempt.directory, "region.wav")))
        import soundfile as sf
        info = sf.info(region_attempt.isolated_path)
        check("region output is about one second, not two",
              0.8 < info.frames / info.samplerate < 1.2,
              f"{info.frames / info.samplerate:.3f}s")
        check("region attempt is recorded as manual",
              region_attempt.origin == "manual_region", region_attempt.origin)

    window.close()
    spin(app, 500)
    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {FAILURES}")
        return 1
    print("the full Phase 4 acceptance chain holds, with the real model")
    shutil.rmtree(ROOT, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
