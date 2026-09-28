"""Phase 5 GUI: theme, formatting, visualisation, controller, and widget wiring.

Runs headless.  ``QT_QPA_PLATFORM=offscreen`` is forced at import time so the
widget tests need no display, which is what makes them runnable in CI and in
this repository's test environment.

The split under test is deliberate: most behaviour lives in Qt-free modules
(``theme``, ``formatting``, ``audioview``, ``controller``) and is tested
directly; the widget tests then check that the window is wired to them and does
not reintroduce the work they already do.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.events.database import Decision, EventDatabase  # noqa: E402
from app.gui.audioview import (  # noqa: E402
    AudioCorrupt,
    AudioUnavailable,
    Envelope,
    build_envelope,
    build_spectrogram,
    ramp_lut,
    read_event_audio,
)
from app.gui.controller import ReviewController, open_controller  # noqa: E402
from app.gui.formatting import (  # noqa: E402
    UNAVAILABLE,
    build_row,
    build_rows,
    describe_event,
    detector_confidence,
    format_decision,
    level,
    rate,
    user_confidence,
)
from app.gui.theme import (  # noqa: E402
    THEME_PACK_DIR,
    build_stylesheet,
    load_theme,
    normalise_color,
)

SR = 48000


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="session")
def app_qt():
    """One QApplication for the whole session; Qt allows only one."""
    from PyQt5.QtWidgets import QApplication

    existing = QApplication.instance()
    if existing is not None:
        yield existing
        return
    application = QApplication([])
    yield application
    application.quit()


@pytest.fixture
def populated(tmp_path):
    """A small database with real events, audio and a fingerprint."""
    from app.audio.wavio import write_wav_atomic
    from app.detection.tracker import (
        EventProfile,
        EventTermination,
        TrackedEvent,
    )
    from app.events.fingerprint import build_fingerprint
    from app.events.event import build_event
    from app.detection.classifier import Classifier

    rng = np.random.default_rng(11)
    audio_dir = tmp_path / "events"
    audio_dir.mkdir()

    def make_profile(centroid, level_db, presence, onsets):
        p = EventProfile(
            onset_sample=0, last_active_sample=SR * 2, frames=200,
            frames_with_features=200, frames_above_floor=int(200 * presence),
            frames_per_second=100.0, peak_snr_db=20.0, mean_snr_db=17.0,
            mean_rms_dbfs=level_db, peak_dbfs=level_db + 8.0,
            max_crest=4.0, mean_crest=2.4, mean_modulation=0.3,
            mean_flatness=0.3, mean_centroid_hz=centroid,
            mean_bandwidth_hz=3000.0, max_flux=0.3,
            mean_low_band_snr_db=10.0, mean_high_band_snr_db=9.0,
            mean_mid_band_snr_db=14.0, high_to_low_ratio_db=-1.0,
            impulsive_onsets=onsets, onset_intervals=[0.5] * max(onsets - 1, 0),
        )
        for i in range(200):
            p.level_moments.add(level_db + (i % 7) / 7.0)
            p.level_series.append(level_db)
            p.snr_moments.add(18.0)
            p.centroid_moments.add(centroid + (i % 7))
            p.flatness_moments.add(0.3)
        return p

    db = EventDatabase(str(tmp_path / "events.db"))
    specs = [
        ("event_000000", 1000.0, -40.0, 0.8, 3, True),
        ("event_000001", 6000.0, -18.0, 0.2, 1, True),
        ("event_000002", 1000.0, -40.0, 0.8, 3, False),   # no audio on disk
    ]
    for event_id, centroid, level_db, presence, onsets, with_audio in specs:
        profile = make_profile(centroid, level_db, presence, onsets)
        tracked = TrackedEvent(
            start_sample=0, end_sample=SR * 2, sample_rate=SR,
            profile=profile, termination=EventTermination.QUIET,
        )
        result = Classifier().classify(profile, 2.0)
        event = build_event(tracked, result, event_id)
        audio_path = None
        if with_audio:
            t = np.arange(SR * 2) / SR
            samples = (
                0.3 * np.sin(2 * np.pi * centroid * t)
                + 0.05 * rng.standard_normal(SR * 2)
            ).astype(np.float32)
            directory = audio_dir / event_id
            directory.mkdir(parents=True, exist_ok=True)
            audio_path = str(directory / "original.wav")
            write_wav_atomic(audio_path, samples, SR)
            # Make the record coherent, as EventStore.extract would: the audio
            # on disk is the audio the metadata claims.
            event.audio_path = audio_path
            event.audio_frames = int(samples.size)
            event.requested_frames = int(samples.size)
        db.store_event(
            event_id=event_id,
            timestamp="2026-09-27T09:41:17+00:00",
            duration=2.0,
            metadata=event.metadata(),
            fingerprint=build_fingerprint(profile, termination="quiet"),
            audio_path=audio_path,
            start_seconds=0.0,
            end_seconds=2.0,
            classification=result.classification,
        )
    yield db
    db.close()


@pytest.fixture
def controller(populated):
    from app.config import AppConfig

    return ReviewController(populated, sample_rate=SR,
                            config=AppConfig().validate())


# ======================================================================
# Theme pack integration
# ======================================================================
def test_theme_pack_is_present():
    assert os.path.isdir(THEME_PACK_DIR), "the design system must ship with the app"
    assert os.path.exists(os.path.join(THEME_PACK_DIR, "css",
                                       "srp-theme.css"))
    assert os.path.exists(os.path.join(THEME_PACK_DIR, "css",
                                       "srp-theme-tokens.css"))


def test_theme_reads_tokens_from_the_pack():
    theme = load_theme("dark")
    assert theme.available
    assert len(theme.tokens) > 30
    # Colours must resolve to Qt-usable hex, not the default.
    assert theme.color("bg").startswith("#")
    assert theme.color("bg") != "#888888"
    assert theme.color("primary") != "#888888"
    assert theme.color("text") != "#888888"


def test_theme_resolves_both_naming_schemes():
    """The pack names colours --srp-bg and --color-bg for the same thing."""
    theme = load_theme("dark")
    assert theme.get("bg")
    assert theme.get("color-bg")
    assert theme.color("bg") == theme.color("color-bg")


def test_theme_metrics_are_not_default():
    theme = load_theme("dark")
    assert theme.px("radius-md") == 12
    assert theme.px("space-4") == 16


def test_light_flavour_keeps_the_base_metrics():
    """The pack's light block overrides colours only; metrics must survive."""
    light = load_theme("light")
    dark = load_theme("dark")
    assert light.color("bg") != dark.color("bg")
    assert light.px("radius-md") == dark.px("radius-md")
    assert light.px("space-4") == dark.px("space-4")


def test_theme_degrades_without_the_pack(tmp_path):
    theme = load_theme("dark", theme_dir=str(tmp_path / "absent"))
    assert theme.available is False
    # A missing design system must not crash the stylesheet builder.
    assert build_stylesheet(theme) == ""


def test_colour_normalisation():
    assert normalise_color("#abc") == "#aabbcc"
    assert normalise_color("#0f0a18") == "#0f0a18"
    assert normalise_color("rgb(18, 240, 18)") == "#12f012"
    assert normalise_color("rgba(18, 240, 18, 0.5)") == "#12f012"
    assert normalise_color("nonsense") == "#888888"
    assert normalise_color("") == "#888888"


def test_stylesheet_uses_theme_colours():
    theme = load_theme("dark")
    sheet = build_stylesheet(theme)
    assert theme.color("bg") in sheet
    assert theme.color("primary") in sheet
    assert len(sheet) > 500


def test_spectral_ramp_comes_from_the_theme():
    ramp = load_theme("dark").spectral_ramp()
    assert len(ramp) >= 4
    assert all(c.startswith("#") for c in ramp)
    lut = ramp_lut(ramp)
    assert lut.shape == (256, 3)


# ======================================================================
# Formatting - human readable, nothing invented
# ======================================================================
def test_decisions_use_the_backend_enum():
    for decision in Decision:
        assert format_decision(decision)
        assert format_decision(decision.value)


def test_missing_level_is_reported_not_invented():
    """The backend's -120 dBFS floor must not display as a real level."""
    assert level(-200.0) == UNAVAILABLE
    assert level(None) == UNAVAILABLE
    assert level("nonsense") == UNAVAILABLE
    assert level(-40.0) == "-40.0 dBFS"


def test_rate_is_shown_in_hz():
    assert rate(48000) == "48 000 Hz"
    assert rate(None) == UNAVAILABLE


def test_detector_confidence_is_not_invented(populated):
    """The backend reports null, so the GUI must not print 0%."""
    assert detector_confidence({}) == "not calibrated"
    assert detector_confidence({"classification": {"confidence": None}}) == (
        "not calibrated"
    )
    # A value that exists is shown, and attributed.
    assert detector_confidence({"classification": {"confidence": 0.73}}) == "73%"


def test_user_confidence_is_separate_from_detector_confidence():
    assert user_confidence(None) == "not stated"
    assert user_confidence(0.8) == "80%"
    assert user_confidence(80) == "80%"


def test_description_sections_populated(controller):
    stored, description = controller.select("event_000000")
    assert description.headline
    assert description.acoustics
    assert description.source
    assert description.review
    assert description.technical
    labels = {f.label for f in description.review}
    assert "Detector confidence" in labels
    assert "Reviewer confidence" in labels


def test_description_includes_a_source_section(controller):
    _stored, description = controller.select("event_000000")
    fields = {f.label: f.value for f in description.source}
    assert "Source" in fields
    assert "Internal rate" in fields
    assert fields["Internal rate"] == "48 000 Hz"


def test_rows_are_compact_by_default(controller):
    rows = build_rows(controller.list_events())
    assert rows
    for row in rows:
        # Event number, time, duration, decision, label - nothing technical.
        assert len(row.compact()) == 5
        assert row.compact()[0] == row.event_id, "the number shown is the id"
        assert row.detail                       # available behind "Details"
        assert len(row.full()) == 10


def test_row_marks_missing_audio(controller):
    rows = {r.event_id: r for r in build_rows(controller.list_events())}
    assert rows["event_000000"].status == "complete"
    assert rows["event_000002"].status == "no audio"


# ======================================================================
# Audio and visualisation
# ======================================================================
def test_envelope_preserves_transients():
    """A mean would flatten this; a min/max envelope must not."""
    samples = np.zeros(4096, dtype=np.float32)
    samples[2048] = 1.0
    envelope = build_envelope(samples, SR, columns=256)
    assert isinstance(envelope, Envelope)
    assert envelope.peak == pytest.approx(1.0, rel=1e-3)
    assert envelope.columns == 256


def test_envelope_is_bounded_in_width():
    for length in (10, 5000, 10_000_000):
        data = np.zeros(length, dtype=np.float32)
        envelope = build_envelope(data, SR, columns=1200)
        assert envelope.columns <= 1200


def test_envelope_empty_input():
    envelope = build_envelope(np.zeros(0, dtype=np.float32), SR)
    assert envelope.columns == 0
    assert envelope.duration == 0.0


def test_spectrogram_shape_and_range():
    t = np.arange(SR) / SR
    samples = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    image = build_spectrogram(samples, SR, columns=128, max_hz=8000.0)
    assert image.bins > 0 and image.columns > 0
    assert image.data.min() >= 0.0 and image.data.max() <= 1.0
    assert image.frequencies[-1] <= 8000.0


def test_spectrogram_handles_empty_input():
    image = build_spectrogram(np.zeros(0, dtype=np.float32), SR)
    assert image.bins == 0


def test_missing_audio_is_distinguishable_from_corrupt(tmp_path):
    with pytest.raises(AudioUnavailable):
        read_event_audio(str(tmp_path / "absent.wav"), SR)
    with pytest.raises(AudioUnavailable):
        read_event_audio(None, SR)
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"RIFFnot-a-wav-file")
    with pytest.raises(AudioCorrupt):
        read_event_audio(str(broken), SR)


def test_audio_resampled_to_the_internal_rate(tmp_path):
    from app.audio.wavio import write_wav_atomic

    path = str(tmp_path / "a.wav")
    t = np.arange(44100) / 44100
    write_wav_atomic(path, (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
                     44100)
    samples = read_event_audio(path, SR)
    assert abs(samples.size - 48000) < 1000


# ======================================================================
# Controller
# ======================================================================
def test_controller_lists_events(controller):
    events = controller.list_events()
    assert len(events) == 3
    assert events[0].event_id == "event_000002"   # newest first


def test_controller_filters_by_decision(controller):
    controller.annotate("event_000001", decision=Decision.REJECTED)
    rejected = controller.list_events(decisions=[Decision.REJECTED])
    assert [e.event_id for e in rejected] == ["event_000001"]


def test_controller_select_returns_the_stored_record(controller):
    stored, description = controller.select("event_000000")
    assert stored.event_id == "event_000000"
    assert stored.fingerprint is not None
    assert description.headline
    assert controller.select("nope") == (None, None)


def test_controller_annotation_persists(controller):
    controller.annotate(
        "event_000000", decision=Decision.CONFIRMED, label="mine",
        confidence=0.9, notes="listened",
    )
    stored = controller.db.get_event("event_000000")
    assert stored.decision is Decision.CONFIRMED
    assert stored.label == "mine"
    assert stored.notes == "listened"
    assert stored.confidence == pytest.approx(0.9)
    # The detector's own measurements are untouched.
    assert stored.fingerprint is not None
    assert stored.metadata["classification"]["label"]


def test_controller_notes_only_edit_keeps_the_decision(controller):
    controller.annotate("event_000000", decision=Decision.SAVED)
    controller.annotate("event_000000", notes="second thoughts")
    stored = controller.db.get_event("event_000000")
    assert stored.decision is Decision.SAVED
    assert stored.notes == "second thoughts"


def test_controller_clear_label_keeps_history(controller):
    controller.annotate("event_000000", decision=Decision.CONFIRMED, label="x")
    controller.clear_label("event_000000")
    stored = controller.db.get_event("event_000000")
    assert stored.label in (None, "")
    assert len(stored.annotations) == 2      # append-only, nothing lost


def test_controller_rejected_events_are_kept(controller):
    controller.annotate("event_000000", decision=Decision.REJECTED)
    assert controller.db.get_event("event_000000") is not None
    assert controller.db.get_event("event_000000").fingerprint is not None


def test_controller_similarity(controller):
    hits = controller.similar("event_000000", limit=3)
    assert hits
    # The acoustically similar event is closer than the different one.
    ids = {event_id: distance for event_id, distance, _ in hits}
    if "event_000002" in ids and "event_000001" in ids:
        assert ids["event_000002"] < ids["event_000001"]


def test_controller_audio_and_envelope_are_cached(controller):
    stored, _ = controller.select("event_000000")
    pending = controller.begin_audio(stored)
    for _ in range(200):
        if pending.done:
            break
        import time
        time.sleep(0.01)
    assert pending.ok
    assert controller.cached_audio("event_000000") is not None

    envelope_pending = controller.begin_envelope(stored)
    for _ in range(200):
        if envelope_pending.done:
            break
        import time
        time.sleep(0.01)
    assert controller.cached_envelope("event_000000") is not None
    assert controller.cache_size() >= 2

    controller.evict("event_000000")
    assert controller.cached_audio("event_000000") is None


def test_controller_missing_audio_does_not_raise(controller):
    stored, _ = controller.select("event_000002")   # no audio on disk
    pending = controller.begin_audio(stored)
    for _ in range(200):
        if pending.done:
            break
        import time
        time.sleep(0.01)
    assert pending.done
    assert not pending.ok
    assert pending.error_kind == "missing"
    # The event is still fully usable.
    assert controller.db.get_event("event_000002") is not None


def test_controller_submit_does_not_block(controller):
    """A decode must not run on the caller's thread."""
    stored, _ = controller.select("event_000000")
    import time
    began = time.monotonic()
    controller.begin_audio(stored)
    assert time.monotonic() - began < 0.5


def test_change_marker_is_cheap_and_detects_a_new_event(controller):
    before = controller.change_marker()
    controller.db.store_event(
        event_id="event_000099", timestamp="t", duration=1.0, metadata={}
    )
    after = controller.change_marker()
    assert after != before
    assert after[0] == before[0] + 1


def test_open_controller_requires_an_existing_database(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_controller(str(tmp_path / "absent.db"))


# ======================================================================
# Widgets
# ======================================================================
def _window(controller, tmp_path):
    from app.gui.main_window import MainWindow
    from app.gui.theme import load_theme

    return MainWindow(controller, load_theme("dark"), db_path=str(tmp_path))


def test_window_builds_and_lists_events(app_qt, controller, tmp_path):
    window = _window(controller, tmp_path)
    try:
        assert window.event_list.list.topLevelItemCount() == 3
        assert window.size().width() > 0
    finally:
        window.close()


def test_selecting_an_event_loads_its_metadata(app_qt, controller, tmp_path):
    window = _window(controller, tmp_path)
    try:
        window.select_event("event_000000")
        assert window._stored.event_id == "event_000000"
        assert "event_000000" in window.title_label.text()
        # Panels are populated.
        for panel in window._panels.values():
            assert panel._rows
        # A source box exists, as required, and is populated.
        assert "Source" in window._panels
        source_labels = [
            child.text()
            for row in window._panels["Source"]._rows
            for child in row.findChildren(type(window._panels["Source"]._rows[0]))
            if hasattr(child, "text") and child.text()
        ]
        assert any("Source" in text for text in source_labels), source_labels
    finally:
        window.close()


def test_selecting_another_event_updates_the_inspector(app_qt, controller,
                                                      tmp_path):
    window = _window(controller, tmp_path)
    try:
        window.select_event("event_000000")
        first = window._stored.event_id
        first_title = window.title_label.text()
        window.select_event("event_000001")
        assert window._stored.event_id != first
        assert window.title_label.text() != first_title
    finally:
        window.close()


def test_annotation_through_the_window_persists(app_qt, controller, tmp_path):
    from app.events.database import Decision as D

    window = _window(controller, tmp_path)
    try:
        window.select_event("event_000000")
        window.review_bar.label_edit.setText("from gui")
        window.review_bar._choose(D.SAVED)
        stored = controller.db.get_event("event_000000")
        assert stored.decision is D.SAVED
        assert stored.label == "from gui"
    finally:
        window.close()


def test_window_tolerates_missing_audio(app_qt, controller, tmp_path):
    window = _window(controller, tmp_path)
    try:
        window.select_event("event_000002")
        import time
        for _ in range(200):
            window._pump_pending()
            if window._pending_audio is None:
                break
            time.sleep(0.01)
        # No crash, transport disabled, message shown.
        assert not window.play_button.isEnabled()
        assert "Audio unavailable" in window.waveform._message
        # Annotation still works.
        window.review_bar.label_edit.setText("no audio here")
        window.review_bar._emit_save()
        assert controller.db.get_event("event_000002").label == "no audio here"
    finally:
        window.close()


def test_spectrogram_is_not_built_until_the_tab_is_opened(app_qt, controller,
                                                          tmp_path):
    import time
    window = _window(controller, tmp_path)
    try:
        window.select_event("event_000000")
        for _ in range(200):
            window._pump_pending()
            if window._pending_audio is None:
                break
            time.sleep(0.01)
        for _ in range(200):
            window._pump_pending()
            if controller.cached_envelope("event_000000") is not None:
                break
            time.sleep(0.01)
        # Nothing built yet.
        assert controller.cached_spectrogram("event_000000") is None
        assert window.spectrogram._image is None
        # Opening the tab builds it.
        for index in range(window.tabs.count()):
            if window.tabs.tabText(index) == "Spectrogram":
                window.tabs.setCurrentIndex(index)
        for _ in range(300):
            window._pump_spectrogram()
            if window.spectrogram._image is not None:
                break
            time.sleep(0.01)
        assert window.spectrogram._image is not None
    finally:
        window.close()


def test_selecting_an_event_does_not_regenerate_the_fingerprint(
        app_qt, controller, tmp_path):
    """The stored fingerprint is authoritative; selecting must not recompute."""
    before = controller.db.get_event("event_000000").fingerprint.to_dict()
    window = _window(controller, tmp_path)
    try:
        for _ in range(3):
            window.select_event("event_000000")
        after = controller.db.get_event("event_000000").fingerprint.to_dict()
        assert before == after
    finally:
        window.close()


def test_live_tick_reloads_only_when_the_database_changed(app_qt, controller,
                                                         tmp_path):
    window = _window(controller, tmp_path)
    window.show()   # live updates are skipped for a hidden window
    app_qt.processEvents()
    try:
        marker = controller.change_marker()
        window._change_marker = marker
        # No change: the list is not rebuilt.
        window.refresh()
        items = window.event_list.list.topLevelItemCount()
        window._on_live_tick()
        assert window.event_list.list.topLevelItemCount() == items
        # A new event appears without a restart.
        controller.db.store_event(
            event_id="event_000500", timestamp="t", duration=1.0, metadata={}
        )
        window._on_live_tick()
        assert window.event_list.list.topLevelItemCount() == items + 1
    finally:
        window.close()


def test_review_bar_uses_backend_decisions(app_qt, controller, tmp_path):
    from app.gui.widgets import ReviewBar

    window = _window(controller, tmp_path)
    try:
        window.select_event("event_000000")
        bar = window.review_bar
        values = {decision.value for _text, decision in bar.BUTTON_DECISIONS}
        assert values <= {d.value for d in Decision}
        assert "saved" in values and "rejected" in values
        # No invented states.
        for banned in ("good", "bad", "real", "fake", "interesting"):
            assert banned not in values
    finally:
        window.close()


def test_gui_modules_contain_no_sound_class_vocabulary():
    """Section 25: the interface must stay application-agnostic."""
    import re
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "app" / "gui"
    pattern = re.compile(
        r"(?<![.\w])(breath(ing|e)?|respirat\w*|snor\w+|sleep\w*|heartbeat|"
        r"footsteps?|knocks?|cough\w*|whispers?|speech|doorbell|"
        r"animals?|machinery|alarms?|music)(?![\w])",
        re.IGNORECASE,
    )
    for name in ("theme.py", "formatting.py", "audioview.py", "controller.py",
                  "widgets.py", "main_window.py", "app.py"):
        offenders = [
            line.strip()
            for line in (base / name).read_text().splitlines()
            if pattern.search(line.split("#", 1)[0])
        ]
        assert not offenders, f"{name}: {offenders}"


# ======================================================================
# The single entry point
# ======================================================================
def test_launcher_defaults_to_the_network_source():
    """The default is the phone on 8190, not the local microphone.

    It was `device`, which silently recorded the built-in mic while the phone
    found nothing listening.
    """
    from app.launch import build_parser

    args = build_parser().parse_args([])
    assert args.source == "network"
    assert args.port == 8190
    assert args.db == "events.db"
    assert args.events == "events"
    assert args.no_capture is False
    assert args.capture_only is False
    # The local input stays available, just not by default.
    assert build_parser().parse_args(["--source", "device"]).source == "device"


def test_launcher_builds_a_capture_command_with_its_settings():
    from app.launch import CaptureProcess, build_parser

    args = build_parser().parse_args([
        "--db", "x.db", "--events", "ev", "--recordings", "rec",
        "--source", "network", "--port", "9000", "--seconds", "30",
    ])
    joined = " ".join(CaptureProcess(args).command())
    assert "-m app.capture" in joined
    assert "--db x.db" in joined
    assert "--events ev" in joined
    assert "--output rec" in joined
    assert "--source network" in joined
    assert "--port 9000" in joined
    assert "--seconds 30" in joined


def test_launcher_uses_this_interpreter_for_its_child():
    """The child must run in the same virtualenv, or its imports fail."""
    import sys

    from app.launch import python_executable

    assert python_executable() == sys.executable


def test_launcher_creates_a_missing_database(tmp_path):
    from app.events.database import EventDatabase
    from app.launch import _ensure_database

    path = str(tmp_path / "fresh.db")
    assert not os.path.exists(path)
    assert _ensure_database(path) is True
    assert os.path.exists(path)
    database = EventDatabase(path)
    try:
        assert database.count() == 0
    finally:
        database.close()
    # Idempotent: a second call is a no-op rather than an error.
    assert _ensure_database(path) is True


def test_entry_point_routes_options_to_the_launcher(capsys):
    """`launch.sh --source network` must not be an unknown command."""
    import app.__main__ as entry

    assert entry.main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "python -m app" in out
    assert "--source network" in out
    # A leading option is a launcher option, not a sub-command.
    source = __import__("pathlib").Path(entry.__file__).read_text()
    assert 'command.startswith("-")' in source


def test_launch_script_forwards_its_arguments():
    """launch.sh used to drop its arguments and run the default database."""
    import pathlib

    script = pathlib.Path(__file__).resolve().parents[1] / "launch.sh"
    source = script.read_text(encoding="utf-8")
    # User arguments are appended after the environment defaults, so they win.
    assert 'ARGS+=("$@")' in source


def test_desktop_entry_points_at_the_launcher():
    """The icon must start the same single entry point, not a variant."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    entry = (root / "audio-microscope.desktop").read_text(encoding="utf-8")
    assert "Exec=" + str(root / "launch.sh") in entry
    assert "Type=Application" in entry
    # The icon is the project's own mark.
    assert "lary.png" in entry
    icon = root / "app/gui/srp-css-theme-pack/assets/lary.png"
    assert icon.exists(), "the icon named by the desktop entry must exist"


# ======================================================================
# Splash: start the stream, then press any key
# ======================================================================
def test_splash_states_the_steps_and_waits_for_a_key(app_qt):
    from app.gui.splash import SplashWindow
    from app.gui.theme import load_theme

    splash = SplashWindow(load_theme("dark"), db_path="events.db", port=8190)
    try:
        assert splash.was_dismissed is False
        texts = splash._step_texts()
        assert any("press any key" in text for text in texts)
        assert any("Start" in text for text in texts)
        assert "8190" in " ".join(splash._step_texts()), "the port must be shown"
        # The splash is dismissed by the gesture the instruction names.
        from PyQt5.QtCore import QEvent, Qt
        from PyQt5.QtGui import QKeyEvent

        splash.keyPressEvent(
            QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.NoModifier)
        )
        assert splash.was_dismissed is True
    finally:
        splash.close()


def test_splash_reads_the_real_counter_from_the_capture_log(app_qt, tmp_path):
    """The counter belongs to the capture process, so it is read from its log."""
    from app.gui.splash import SplashWindow
    from app.gui.theme import load_theme

    log = tmp_path / "capture.log"
    log.write_text(
        "  listening on port 8190\n"
        "  client connected: 10.0.0.242:34118\n"
        "    10.0.0.242:34118: 12.4s received, 4 config key(s)\n",
        encoding="utf-8",
    )
    splash = SplashWindow(load_theme("dark"), capture_log=str(log))
    try:
        assert splash.poll_capture_log() == (12.4, True)
    finally:
        splash.close()

    # Nothing received yet, and no log at all, are both "waiting" - never a
    # number invented to fill the gap.
    empty = tmp_path / "empty.log"
    empty.write_text("  listening on port 8190\n", encoding="utf-8")
    waiting = SplashWindow(load_theme("dark"), capture_log=str(empty))
    try:
        assert waiting.poll_capture_log() == (0.0, False)
    finally:
        waiting.close()

    absent = SplashWindow(load_theme("dark"), capture_log=str(tmp_path / "gone.log"))
    try:
        assert absent.poll_capture_log() == (0.0, False)
    finally:
        absent.close()


def test_splash_reports_no_capture_when_there_is_no_log(app_qt):
    from app.gui.splash import SplashWindow
    from app.gui.theme import load_theme

    splash = SplashWindow(load_theme("dark"), capture_log=None)
    try:
        assert "capture is not running" in splash._capture_note.text()
    finally:
        splash.close()


def test_launcher_can_skip_the_splash():
    from app.launch import build_parser

    assert build_parser().parse_args([]).no_splash is False
    assert build_parser().parse_args(["--no-splash"]).no_splash is True


# ======================================================================
# The events tree the controller is told about
# ======================================================================
def test_controller_reads_the_events_root_it_was_given(tmp_path):
    """Separation results live in the tree capture wrote, not the project's.

    A run with ``--events /somewhere/else`` stores its event directories
    there.  If the controller looked in the project's own ``events`` directory
    it would not find the separations it made, and - worse - a search by
    event id could match an unrelated event in the project tree and report
    that event's attempts instead.
    """
    from app.config import AppConfig
    from app.separation.store import SeparationStore

    root = tmp_path / "elsewhere"
    db = EventDatabase(str(tmp_path / "e.db"))
    store = SeparationStore(str(root))

    controller = ReviewController(
        db, sample_rate=SR, config=AppConfig().validate(),
        events_root=str(root),
    )
    assert controller._separation_root() == str(root)

    # An attempt recorded in that tree is found.
    attempt_dir = store.allocate_attempt(str(root / "2026-09-27" / "event_000001"))
    store.write_metadata(attempt_dir, {
        "query": "a query", "status": "succeeded", "input_seconds": 1.0,
        "processing_seconds": 4.0, "realtime_ratio": 4.0,
    })
    db.store_event(
        event_id="event_000001", timestamp="2026-09-27T09:41:17+00:00",
        duration=2.0, metadata={}, audio_path=None,
        classification="probe",
    )
    attempts = controller.separation_attempts(db.get_event("event_000001"))
    assert [a.name for a in attempts] == ["separation_001"]
    assert attempts[0].query == "a query"
    assert attempts[0].realtime_ratio == 4.0
    db.close()


def test_controller_defaults_to_the_project_events_directory(tmp_path):
    from app.config import AppConfig

    # A real temp path, not ":memory:": EventDatabase applies abspath() to it,
    # which turns that string into a file named ":memory:" in the project root.
    controller = ReviewController(
        EventDatabase(str(tmp_path / "e.db")), sample_rate=SR,
        config=AppConfig().validate(),
    )
    # This file is tests/test_gui.py, so the project root is two levels up.
    expected = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "events",
    )
    assert controller._separation_root() == expected


def test_the_model_is_found_regardless_of_the_events_root(tmp_path):
    """The model lives in the project; the events tree does not have to.

    Conflating the two roots makes separation unavailable whenever capture
    wrote its events anywhere else, because the model paths stop resolving.
    """
    from app.config import AppConfig

    controller = ReviewController(
        EventDatabase(str(tmp_path / "e.db")), sample_rate=SR,
        config=AppConfig().validate(), events_root=str(tmp_path / "elsewhere"),
    )
    assert controller._project_root != controller._separation_root()
    assert os.path.isdir(
        os.path.join(controller._project_root, "third_party", "AudioSep")
    )


def test_capture_cli_defaults_to_the_network_source():
    """Same reasoning as the launcher: the phone on 8190, not the mic.

    `app.capture` was left defaulting to `device` after the launcher had
    already been corrected.  On this machine `--source device` resolves to
    the HDA Intel PCH ALC897 analog path, which only carries audio while
    scrcpy is running - so a bare `python -m app.capture` recorded silence
    and reported it as a quiet environment.
    """
    from app.capture import build_parser

    args = build_parser().parse_args([])
    assert args.source == "network"
    assert args.port == 8190
    # The local input stays available, just not by default.
    assert build_parser().parse_args(["--source", "device"]).source == "device"


# ======================================================================
# Selecting a region and extracting it as its own event
# ======================================================================
def test_the_window_exposes_region_selection_and_extraction(app_qt, controller):
    """The controls must exist and be wired, not merely implemented.

    The feature is only useful if the operator can reach it: a Select toggle
    on the transport, a region on both views, and an Extract button that
    enables only once a region exists.
    """
    from app.gui.main_window import MainWindow
    from app.gui.theme import load_theme

    window = MainWindow(controller, load_theme("dark"))
    try:
        assert window.extract_button is not None
        assert window.extract_button.isEnabled() is False
        # Selection starts disarmed, so a click still seeks.
        assert window.waveform.selection_enabled is False
        assert window.spectrogram.selection_enabled is False

        window._on_select_region_toggled(True)
        assert window.waveform.selection_enabled is True
        assert window.spectrogram.selection_enabled is True
        # The separation panel shares the same toggle.
        assert window.separation._use_region is True

        # With no event loaded there is no timeline to measure a region on.
        window._event_duration = 0.0
        window._on_region_changed(0.25, 0.75)
        assert window._region is None

        window._event_duration = 8.0
        window._on_region_changed(0.25, 0.75)
        assert window._region == (2.0, 6.0)
        # Both views show it, and the label says what was selected.
        assert window.waveform.region() == (0.25, 0.75)
        assert window.spectrogram.region() == (0.25, 0.75)
        assert "2.00-6.00s" in window.selection_label.text()

        # A collapsed drag clears everything.
        window._on_region_changed(0.0, 0.0)
        assert window._region is None
        assert window.waveform.region() is None
        assert window.extract_button.isEnabled() is False
    finally:
        window.close()
        controller.db.close()


def test_extraction_without_audio_is_refused(app_qt, controller, populated):
    """An event whose audio was evicted cannot be extracted, and says so."""
    from app.gui.main_window import MainWindow
    from app.gui.theme import load_theme

    window = MainWindow(controller, load_theme("dark"))
    try:
        # A row with no audio: the fingerprint is still there.
        event_id = populated.list_events()[0].event_id
        populated.set_audio_path(event_id, None)
        populated.set_audio_state(event_id, available=False)
        window.select_event(event_id)
        window._event_duration = 2.0
        window._on_region_changed(0.1, 0.9)
        window._on_extract_selection()
        message = window.status_label.text()
        assert "no audio" in message.lower()
        # Nothing was queued, because there was nothing to do.
        assert window._pending_extract is None
    finally:
        window.close()
        controller.db.close()


# ======================================================================
# Real mouse interaction
# ======================================================================
@pytest.fixture
def theme():
    from app.gui.theme import load_theme

    return load_theme("dark")


def _drag(widget, from_fraction, to_fraction):
    """A real press-move-release across the widget, as a hand would do it."""
    from PyQt5.QtCore import Qt, QPoint
    from PyQt5.QtTest import QTest

    y = widget.height() // 2
    QTest.mousePress(
        widget, Qt.LeftButton,
        pos=QPoint(int(from_fraction * widget.width()), y))
    QTest.mouseMove(
        widget, QPoint(int(to_fraction * widget.width()), y))
    QTest.mouseRelease(
        widget, Qt.LeftButton,
        pos=QPoint(int(to_fraction * widget.width()), y))


def test_a_real_mouse_drag_selects_a_region(app_qt, theme):
    """Selection has to work with an actual mouse, not just a method call.

    Every mouse handler once used ``QMouseEvent.position()``, which is Qt 6.
    This is PyQt5, where it is ``pos()``; Qt swallowed the AttributeError, so
    a drag did nothing at all and no error was raised anywhere. Calling the
    handler directly cannot catch that, so the drag is performed here.
    """
    from PyQt5.QtWidgets import QApplication
    from app.gui.audioview import Envelope
    from app.gui.widgets import WaveformWidget

    import numpy as np

    application = QApplication.instance()
    widget = WaveformWidget(theme)
    t = np.linspace(0, 1, 4000)
    signal = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    widget.set_envelope(Envelope(
        peaks=np.abs(signal[::10]).astype(np.float32),
        troughs=-np.abs(signal[::10]).astype(np.float32),
        sample_rate=48000, samples=signal.size, peak=0.4,
    ))
    widget.resize(800, 200)
    widget.show()
    application.processEvents()

    seen = []
    widget.region_changed.connect(lambda a, b: seen.append((a, b)))

    # Unarmed, a drag must not select: a click still seeks.
    widget.set_selection_enabled(False)
    _drag(widget, 0.25, 0.75)
    application.processEvents()
    assert widget.region() is None

    widget.set_selection_enabled(True)
    _drag(widget, 0.25, 0.75)
    application.processEvents()
    assert widget.region() is not None, "the drag selected nothing"
    start, end = widget.region()
    assert abs(start - 0.25) < 0.02 and abs(end - 0.75) < 0.02
    assert seen == [(start, end)]

    # A bare click clears the selection rather than leaving a stale region.
    _drag(widget, 0.5, 0.5)
    application.processEvents()
    assert widget.region() is None
    widget.close()


def test_the_spectrogram_selects_by_mouse_too(app_qt, theme):
    from PyQt5.QtWidgets import QApplication
    from app.gui.audioview import Spectrogram
    from app.gui.widgets import SpectrogramWidget

    import numpy as np

    application = QApplication.instance()
    widget = SpectrogramWidget(theme)
    widget.set_spectrogram(Spectrogram(
        data=np.random.default_rng(5).random((64, 200)).astype(np.float32),
        frequencies=np.linspace(0.0, 8000.0, 64),
        sample_rate=48000, max_hz=8000.0,
    ))
    widget.resize(800, 200)
    widget.show()
    application.processEvents()

    widget.set_selection_enabled(True)
    _drag(widget, 0.2, 0.6)
    application.processEvents()
    assert widget.region() is not None
    start, end = widget.region()
    assert abs(start - 0.2) < 0.02 and abs(end - 0.6) < 0.02
    widget.close()


def test_a_click_seeks_with_a_real_mouse(app_qt, theme):
    """Click-to-seek shares the same handler, and had the same bug."""
    from PyQt5.QtCore import Qt, QPoint
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from app.gui.audioview import Envelope
    from app.gui.widgets import WaveformWidget

    import numpy as np

    application = QApplication.instance()
    widget = WaveformWidget(theme)
    t = np.linspace(0, 1, 4000)
    signal = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    widget.set_envelope(Envelope(
        peaks=np.abs(signal[::10]).astype(np.float32),
        troughs=-np.abs(signal[::10]).astype(np.float32),
        sample_rate=48000, samples=signal.size, peak=0.4,
    ))
    widget.resize(800, 200)
    widget.show()
    application.processEvents()

    seeks = []
    widget.seek_requested.connect(seeks.append)
    QTest.mouseClick(widget, Qt.LeftButton,
                     pos=QPoint(400, widget.height() // 2))
    application.processEvents()
    assert seeks, "a click produced no seek"
    assert abs(seeks[-1] - 0.5) < 0.02
    widget.close()


# ======================================================================
# The three review affordances
# ======================================================================
def test_the_event_number_is_in_the_list(app_qt, controller):
    """The id is the handle for everything else, so it must be readable."""
    from app.gui.formatting import build_rows
    from app.gui.widgets import LIST_COLUMNS, EventListWidget

    rows = build_rows(controller.list_events())
    assert rows
    assert "Event" in LIST_COLUMNS
    for row in rows:
        compact = row.compact()
        assert compact[0] == row.event_id
    # A width for every column, or the last one is unset and the header
    # elides.
    assert len(EventListWidget._WIDTHS) >= len(LIST_COLUMNS)


def test_a_similar_hit_opens_that_event(app_qt, controller, populated):
    """The whole point of the panel: a hit you can act on."""
    from app.gui.main_window import MainWindow
    from app.gui.theme import load_theme

    events = populated.list_events()
    if len(events) < 2:
        pytest.skip("needs two stored events")
    window = MainWindow(controller, load_theme("dark"))
    try:
        first = events[0].event_id
        window.select_event(first)
        for _ in range(20):
            app_qt.processEvents()
        hits = controller.similar(first, limit=5)
        if not hits:
            pytest.skip("these events have no comparable fingerprints")
        window._build_similar()
        target = hits[0][0]
        assert window.similar.hits(), "the panel listed nothing"
        assert window.similar.list.topLevelItemCount() == len(hits)
        # Activating a row brings that event into the window.
        window.similar.event_activated.emit(target)
        for _ in range(30):
            app_qt.processEvents()
        assert window._stored.event_id == target
        # And it is named in the status line, so the jump is not a mystery.
        assert target in window.status_label.text()
    finally:
        window.close()
        controller.db.close()


def test_similar_rows_mark_events_whose_audio_is_gone(app_qt, controller,
                                                     populated):
    from app.gui.main_window import MainWindow
    from app.gui.theme import load_theme

    events = populated.list_events()
    if len(events) < 2:
        pytest.skip("needs two stored events")
    # Take the audio away from one event, so a row has to say so.
    victim = events[1].event_id
    populated.set_audio_path(victim, None)
    populated.set_audio_state(victim, available=False)

    window = MainWindow(controller, load_theme("dark"))
    try:
        first = events[0].event_id
        hits = controller.similar(first, limit=5)
        if not hits:
            pytest.skip("no comparable fingerprints")
        window.similar.set_hits(hits)
        texts = [
            window.similar.list.topLevelItem(i).text(3)
            for i in range(window.similar.list.topLevelItemCount())
        ]
        assert any("no audio" in t for t in texts), texts
    finally:
        window.close()
        controller.db.close()


def test_back_returns_to_the_event_an_extraction_came_from(app_qt, controller):
    from app.gui.main_window import MainWindow
    from app.gui.theme import load_theme

    window = MainWindow(controller, load_theme("dark"))
    try:
        assert window.back_button.isVisible() is False
        # Extraction from event_000000 opened event_000001.
        window.select_event("event_000001")
        window._extract_parent_id = "event_000000"
        window.back_button.setText("Back to event_000000")
        window.back_button.setVisible(True)

        window._on_back_to_parent()
        for _ in range(30):
            app_qt.processEvents()
        assert window._stored.event_id == "event_000000"
        # And the way back is gone, because there is nothing to go back from.
        assert window._extract_parent_id is None
        assert window.back_button.isVisible() is False
    finally:
        window.close()
        controller.db.close()


def test_back_does_nothing_when_there_is_no_parent(app_qt, controller):
    from app.gui.main_window import MainWindow
    from app.gui.theme import load_theme

    window = MainWindow(controller, load_theme("dark"))
    try:
        window.select_event("event_000000")
        before = window._stored.event_id
        window._on_back_to_parent()
        for _ in range(20):
            app_qt.processEvents()
        assert window._stored.event_id == before
        assert "no extracted selection" in window.status_label.text().lower()
    finally:
        window.close()
        controller.db.close()


def test_selecting_another_event_clears_the_way_back(app_qt, controller):
    """A Back button that survives an unrelated selection is a trap."""
    from app.gui.main_window import MainWindow
    from app.gui.theme import load_theme

    window = MainWindow(controller, load_theme("dark"))
    try:
        window.select_event("event_000001")
        window._extract_parent_id = "event_000000"
        window.back_button.setVisible(True)
        window.select_event("event_000002")
        assert window._extract_parent_id is None
        assert window.back_button.isVisible() is False
    finally:
        window.close()
        controller.db.close()
