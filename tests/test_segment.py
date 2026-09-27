"""Extracting a selected region of an event as an event of its own.

The scenario these tests exist for: one long event containing the sound worth
keeping *and* the traffic going past and somebody honking, where the operator
wants to keep only the first.  So the assertions are about the operator's
outcome - did the click survive, is it alone in its own event, and is the
original still intact - rather than about internal plumbing.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from app.audio.wavio import read_wav, write_wav_atomic
from app.config import AppConfig
from app.events.database import EventDatabase
from app.events.event import Event
from app.events.segment import (
    SegmentError,
    cut_selection,
    extract_selection,
    next_free_index,
)
from app.events.store import EventStore

SAMPLE_RATE = 48_000


# ----------------------------------------------------------------------
# A parent event shaped like the real problem
# ----------------------------------------------------------------------
def noisy_event(seconds: float = 10.0) -> np.ndarray:
    """Room tone, a car going past, a horn, and a click between them."""
    rng = np.random.default_rng(11)
    total = int(seconds * SAMPLE_RATE)
    audio = (0.0004 * rng.standard_normal(total)).astype(np.float32)

    def add(start: float, length: float, make) -> None:
        begin = int(start * SAMPLE_RATE)
        n = int(length * SAMPLE_RATE)
        if begin + n > total:
            n = total - begin
        t = np.arange(n) / SAMPLE_RATE
        audio[begin:begin + n] += make(t).astype(np.float32)

    # A car: low rumble, 1.0 s to 4.0 s
    add(1.0, 3.0, lambda t: 0.18 * np.sin(2 * np.pi * 70 * t) * np.exp(-t * 0.4))
    # A horn: two loud tones, 6.0 s to 7.0 s
    add(6.0, 0.5, lambda t: 0.5 * np.sin(2 * np.pi * 440 * t))
    add(6.7, 0.5, lambda t: 0.5 * np.sin(2 * np.pi * 440 * t))
    # The click: 40 ms, 8.0 s, on its own
    add(8.0, 0.04, lambda t: 0.4 * np.sin(2 * np.pi * 1800 * t)
        * np.exp(-t * 300))
    return audio


@pytest.fixture
def parent(tmp_path):
    """A real parent event: real audio on disk, and a real database row."""
    root = str(tmp_path / "events")
    os.makedirs(root, exist_ok=True)
    store = EventStore(root)
    audio = noisy_event()
    event = Event(
        event_id="event_000000", start_seconds=0.0, end_seconds=10.0,
        sample_rate=SAMPLE_RATE, classification="probe",
    )
    store.save(event, audio)

    database_path = str(tmp_path / "events.db")
    database = EventDatabase(database_path)
    database.store_event(
        event_id=event.event_id, timestamp=event.detected_at,
        duration=event.duration, metadata=event.metadata(),
        audio_path=event.audio_path, classification="probe",
    )
    size = os.path.getsize(event.audio_path)
    database.set_audio_state(event.event_id, audio_bytes=size, available=True)
    database.set_audio_path(event.event_id, event.audio_path)
    database.close()
    return {
        "root": root,
        "event_id": event.event_id,
        "audio_path": event.audio_path,
        "database": database_path,
        "bytes": size,
    }


def _config() -> AppConfig:
    return AppConfig().validate()


# ======================================================================
# Cutting
# ======================================================================
class TestCut:
    def test_cuts_exactly_the_requested_span(self, parent, tmp_path):
        target = str(tmp_path / "cut.wav")
        samples, rate = cut_selection(
            parent["audio_path"], 7.9, 8.3, target
        )
        assert rate == SAMPLE_RATE
        assert abs(samples.size / rate - 0.4) < 0.01
        assert os.path.exists(target)

    def test_the_cut_contains_the_click(self, parent, tmp_path):
        target = str(tmp_path / "click.wav")
        cut, rate = cut_selection(parent["audio_path"], 7.9, 8.3, target)
        # The click region is far louder than the room tone around it.
        assert float(np.max(np.abs(cut))) > 0.1
        whole = read_wav(parent["audio_path"])
        quiet = whole[int(4.5 * SAMPLE_RATE):int(5.5 * SAMPLE_RATE)]
        assert float(np.max(np.abs(quiet))) < float(np.max(np.abs(cut)))

    def test_the_parent_is_untouched(self, parent, tmp_path):
        before = open(parent["audio_path"], "rb").read()
        cut_selection(parent["audio_path"], 1.0, 2.0,
                      str(tmp_path / "x.wav"))
        assert open(parent["audio_path"], "rb").read() == before

    def test_a_selection_past_the_end_is_refused(self, parent, tmp_path):
        with pytest.raises(SegmentError):
            cut_selection(parent["audio_path"], 30.0, 31.0,
                          str(tmp_path / "x.wav"))

    def test_an_empty_selection_is_refused(self, parent, tmp_path):
        with pytest.raises(SegmentError):
            cut_selection(parent["audio_path"], 2.0, 2.0,
                          str(tmp_path / "x.wav"))

    def test_a_reversed_selection_is_refused(self, parent, tmp_path):
        with pytest.raises(SegmentError):
            cut_selection(parent["audio_path"], 5.0, 4.0,
                          str(tmp_path / "x.wav"))

    def test_a_selection_is_clamped_to_the_end(self, parent, tmp_path):
        samples, rate = cut_selection(
            parent["audio_path"], 9.5, 60.0, str(tmp_path / "end.wav")
        )
        assert 0.4 < samples.size / rate < 0.6

    def test_missing_audio_is_reported(self, tmp_path):
        with pytest.raises(SegmentError):
            cut_selection(str(tmp_path / "nothing.wav"), 0.0, 1.0,
                          str(tmp_path / "x.wav"))


# ======================================================================
# Extraction
# ======================================================================
class TestExtract:
    def test_a_selection_becomes_a_stored_event(self, parent):
        result = extract_selection(
            _config(), parent["event_id"], parent["audio_path"],
            7.9, 8.4, parent["root"], parent["database"],
        )
        assert result.ok, result.error
        assert result.stored_events
        assert result.parent_event_id == parent["event_id"]
        assert 0.4 < result.duration_seconds < 0.6

    def test_the_new_event_has_its_own_audio(self, parent):
        result = extract_selection(
            _config(), parent["event_id"], parent["audio_path"],
            7.9, 8.4, parent["root"], parent["database"],
        )
        database = EventDatabase(parent["database"])
        try:
            stored = database.get_event(result.stored_events[0])
            assert stored is not None
            assert stored.audio_path and os.path.exists(stored.audio_path)
            # Its own audio: inside the selection, never the whole parent.
            assert 0.0 < stored.duration <= result.duration_seconds + 0.02
            assert stored.duration < 1.0
        finally:
            database.close()

    def test_the_provenance_records_the_parent_and_the_region(self, parent):
        result = extract_selection(
            _config(), parent["event_id"], parent["audio_path"],
            7.9, 8.4, parent["root"], parent["database"],
        )
        database = EventDatabase(parent["database"])
        try:
            stored = database.get_event(result.stored_events[0])
            derivation = stored.metadata.get("derivation") or {}
            assert derivation["derived_from_event_id"] == parent["event_id"]
            assert derivation["selection_start_seconds"] == pytest.approx(7.9)
            assert derivation["selection_end_seconds"] == pytest.approx(8.4)
        finally:
            database.close()

    def test_the_parent_is_never_modified(self, parent):
        before = open(parent["audio_path"], "rb").read()
        extract_selection(
            _config(), parent["event_id"], parent["audio_path"],
            7.9, 8.4, parent["root"], parent["database"],
        )
        assert open(parent["audio_path"], "rb").read() == before
        assert os.path.getsize(parent["audio_path"]) == parent["bytes"]

    def test_the_traffic_is_not_in_the_extraction(self, parent):
        # The whole point: a 3 s car and two horns are in the parent, and
        # neither may appear in what was extracted.
        database = EventDatabase(parent["database"])
        try:
            result = extract_selection(
                _config(), parent["event_id"], parent["audio_path"],
                7.9, 8.4, parent["root"], parent["database"],
            )
            stored = database.get_event(result.stored_events[0])
        finally:
            database.close()
        extracted = read_wav(stored.audio_path)
        seconds = extracted.size / SAMPLE_RATE
        assert seconds < 1.0, f"{seconds:.2f}s is not just the click"
        assert float(np.max(np.abs(extracted))) > 0.1
        # The result reports what it actually stored, so the window can tell
        # the operator it is shorter than what they drew.
        assert result.stored_durations
        assert result.stored_durations[0] <= result.duration_seconds + 0.02

    def test_an_event_id_collision_is_impossible(self, parent):
        # Two extractions must not overwrite each other, nor the parent.
        first = extract_selection(
            _config(), parent["event_id"], parent["audio_path"],
            7.9, 8.4, parent["root"], parent["database"],
        )
        second = extract_selection(
            _config(), parent["event_id"], parent["audio_path"],
            7.9, 8.4, parent["root"], parent["database"],
        )
        assert first.stored_events[0] != second.stored_events[0]
        assert os.path.exists(first.stored_events[0] and
                              EventDatabase(parent["database"])
                              .get_event(first.stored_events[0]).audio_path)

    def test_a_very_short_selection_is_stored_even_without_detection(
        self, parent
    ):
        # 40 ms: below the detector's minimum duration, so it would otherwise
        # produce nothing and the operator would lose the very thing they
        # selected.  Stored verbatim, with no invented classification.
        result = extract_selection(
            _config(), parent["event_id"], parent["audio_path"],
            8.0, 8.04, parent["root"], parent["database"],
        )
        assert result.ok, result.error
        assert result.stored_events
        database = EventDatabase(parent["database"])
        try:
            stored = database.get_event(result.stored_events[0])
            assert stored is not None
            # The audio is there, which is what matters.
            assert stored.audio_path and os.path.exists(stored.audio_path)
            samples = read_wav(stored.audio_path)
            assert samples.size > 0
            if result.stored_without_detection:
                # No classification and no fingerprint, because none was
                # measured.
                assert stored.fingerprint is None
                assert stored.classification == "unclassified"
        finally:
            database.close()

    def test_a_selection_with_no_audio_is_reported_not_stored(self, parent):
        # The parent had its audio evicted by retention: the fingerprint is
        # still there, but there is nothing to extract from.
        os.remove(parent["audio_path"])
        result = extract_selection(
            _config(), parent["event_id"], parent["audio_path"],
            1.0, 2.0, parent["root"], parent["database"],
        )
        assert not result.ok
        assert result.error

    def test_progress_is_reported(self, parent):
        seen = []
        extract_selection(
            _config(), parent["event_id"], parent["audio_path"],
            7.9, 8.4, parent["root"], parent["database"],
            on_progress=seen.append,
        )
        assert seen, "the caller should be able to show progress"

    def test_the_result_reports_its_own_cost(self, parent):
        result = extract_selection(
            _config(), parent["event_id"], parent["audio_path"],
            7.9, 8.4, parent["root"], parent["database"],
        )
        assert result.processing_seconds >= 0.0
        assert "stored_events" in result.to_dict()


class TestNextFreeIndex:
    def test_starts_at_zero_for_an_empty_tree(self, tmp_path):
        assert next_free_index(str(tmp_path)) == 0

    def test_steps_past_what_is_there(self, tmp_path):
        for index in (0, 1, 2):
            os.makedirs(str(tmp_path / "2026-01-01" / f"event_{index:06d}"))
        assert next_free_index(str(tmp_path)) == 3

    def test_ignores_directories_that_are_not_events(self, tmp_path):
        os.makedirs(str(tmp_path / "2026-01-01" / "event_000004"))
        os.makedirs(str(tmp_path / "2026-01-01" / "separations"))
        assert next_free_index(str(tmp_path)) == 5
