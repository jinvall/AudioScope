"""Clearing the audio of unreviewed events.

The property under test is not "the files went away". It is that the
*distinction* survives: audio is disposable, the record is not. Every test here
exists to check that a bulk operation over hundreds of events did not quietly
take something it was not supposed to.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from app.config import AppConfig
from app.events.clear import (
    clear_unreviewed_audio,
    unreviewed_with_audio,
)
from app.events.database import Decision, EventDatabase
from app.events.event import Event
from app.events.store import EventStore

SAMPLE_RATE = 48_000


def _fingerprint(value: float = 0.5):
    """A real fingerprint, so "the fingerprint survived" tests something."""
    from app.events.fingerprint import EventFingerprint

    return EventFingerprint.from_dict({
        "fingerprint_version": 1,
        "vector": [value] * 16,
    })


def _event(db_path, root, event_id, seconds=1.0, annotate=None,
           distance=0.5):
    """A real event: real audio on disk, a real row, a real fingerprint."""
    store = EventStore(root)
    samples = (0.3 * np.sin(
        2 * np.pi * 220 * np.linspace(0, seconds, int(seconds * SAMPLE_RATE),
                                       endpoint=False)
    )).astype(np.float32)
    event = Event(event_id=event_id, start_seconds=0.0, end_seconds=seconds,
                  sample_rate=SAMPLE_RATE, classification="probe")
    store.save(event, samples)
    db = EventDatabase(db_path)
    db.store_event(event_id=event_id, timestamp=event.detected_at,
                   duration=event.duration, metadata=event.metadata(),
                   fingerprint=_fingerprint(distance),
                   audio_path=event.audio_path,
                   classification="probe")
    size = os.path.getsize(event.audio_path)
    db.set_audio_state(event_id, audio_bytes=size, available=True)
    db.set_audio_path(event_id, event.audio_path)
    if annotate:
        db.annotate(event_id, annotate)
    db.close()
    return event.audio_path


@pytest.fixture
def world(tmp_path):
    """Four events: three unreviewed, one saved, one rejected."""
    root = str(tmp_path / "events")
    os.makedirs(root, exist_ok=True)
    db_path = str(tmp_path / "events.db")
    paths = {
        "event_000000": _event(db_path, root, "event_000000"),
        "event_000001": _event(db_path, root, "event_000001"),
        "event_000002": _event(db_path, root, "event_000002"),
        "event_000003": _event(db_path, root, "event_000003",
                               annotate=Decision.SAVED),
        "event_000004": _event(db_path, root, "event_000004",
                               annotate=Decision.REJECTED),
    }
    return db_path, root, paths


# ======================================================================
# Selection
# ======================================================================
class TestSelection:
    def test_only_unreviewed_with_audio_are_selected(self, world):
        db_path, root, _paths = world
        db = EventDatabase(db_path)
        try:
            selected = {e.event_id for e in unreviewed_with_audio(db)}
        finally:
            db.close()
        assert selected == {"event_000000", "event_000001", "event_000002"}

    def test_judged_events_cannot_be_selected(self, world):
        """Saved and rejected events are out of reach, by construction."""
        db_path, root, _paths = world
        db = EventDatabase(db_path)
        try:
            selected = {e.event_id for e in unreviewed_with_audio(db)}
        finally:
            db.close()
        assert "event_000003" not in selected, "a saved event was selectable"
        assert "event_000004" not in selected, "a rejected event was selectable"


# ======================================================================
# The preview
# ======================================================================
class TestPreview:
    def test_a_preview_changes_nothing(self, world):
        db_path, root, paths = world
        db = EventDatabase(db_path)
        try:
            report = clear_unreviewed_audio(db, dry_run=True)
        finally:
            db.close()
        assert report.dry_run is True
        assert report.with_audio == 3
        assert report.freed_bytes > 0
        # Every file is still there, and every row still claims its audio.
        for path in paths.values():
            assert os.path.exists(path)
        db = EventDatabase(db_path)
        try:
            for event in db.list_events():
                assert event.audio_available is True
        finally:
            db.close()

    def test_the_preview_reports_what_the_run_does(self, world):
        """The figures confirmed must be the figures acted on."""
        db_path, root, _paths = world
        db = EventDatabase(db_path)
        try:
            preview = clear_unreviewed_audio(db, dry_run=True)
        finally:
            db.close()
        db = EventDatabase(db_path)
        try:
            run = clear_unreviewed_audio(db, dry_run=False)
        finally:
            db.close()
        assert run.with_audio == preview.with_audio
        assert run.freed_bytes == preview.freed_bytes
        assert run.removed == preview.removed

    def test_nothing_to_do_is_reported_as_such(self, tmp_path):
        db = EventDatabase(str(tmp_path / "empty.db"))
        try:
            report = clear_unreviewed_audio(db, dry_run=True)
            assert report.would_change_anything is False
            assert "No unreviewed event" in report.summary()
        finally:
            db.close()

    def test_the_summary_names_the_size_and_the_count(self, world):
        db_path, root, _paths = world
        db = EventDatabase(db_path)
        try:
            report = clear_unreviewed_audio(db, dry_run=True)
        finally:
            db.close()
        summary = report.summary()
        assert "3" in summary and "unreviewed" in summary
        assert "MB" in summary or "GB" in summary or "KB" in summary


# ======================================================================
# The run
# ======================================================================
class TestRun:
    def test_the_audio_goes(self, world):
        db_path, root, paths = world
        db = EventDatabase(db_path)
        try:
            clear_unreviewed_audio(db, dry_run=False)
        finally:
            db.close()
        for event_id in ("event_000000", "event_000001", "event_000002"):
            assert not os.path.exists(paths[event_id]), event_id

    def test_the_fingerprints_survive(self, world):
        """The whole point. A cleared event is still comparable."""
        db_path, root, _paths = world
        db = EventDatabase(db_path)
        try:
            clear_unreviewed_audio(db, dry_run=False)
        finally:
            db.close()
        db = EventDatabase(db_path)
        try:
            events = {e.event_id: e for e in db.list_events()}
            assert len(events) == 5, "an event was deleted"
            for event_id in ("event_000000", "event_000001", "event_000002"):
                stored = events[event_id]
                assert stored.fingerprint is not None, event_id
                assert stored.metadata, event_id
                assert stored.duration is not None
        finally:
            db.close()

    def test_a_cleared_event_is_still_listable_and_searchable(self, world):
        db_path, root, _paths = world
        db = EventDatabase(db_path)
        try:
            clear_unreviewed_audio(db, dry_run=False)
            # Still in the listing: nothing was removed from the record.
            assert len(db.list_events()) == 5
            # Judged events are still filterable by their decision.
            saved = db.list_events(decisions=[Decision.SAVED])
            assert [e.event_id for e in saved] == ["event_000003"]
            # And a cleared event is still reachable by fingerprint search,
            # which is what "searchable" means in this system: the audio went,
            # the description of the sound did not.
            target = db.get_event("event_000000")
            assert target.fingerprint is not None
            found = db.find_similar(target.fingerprint, limit=5)
            assert any(event_id == "event_000000" for event_id, _ in found)
        finally:
            db.close()

    def test_a_cleared_event_does_not_claim_playable_audio(self, world):
        """Otherwise the interface offers playback that cannot work."""
        db_path, root, _paths = world
        db = EventDatabase(db_path)
        try:
            clear_unreviewed_audio(db, dry_run=False)
            for event in db.list_events():
                if event.decision.value == "unreviewed":
                    assert event.audio_available is False
                    assert event.audio_path is None
                    assert event.is_playable is False
                    assert event.audio_evicted_at is not None
        finally:
            db.close()

    def test_judged_events_keep_their_audio(self, world):
        db_path, root, paths = world
        db = EventDatabase(db_path)
        try:
            clear_unreviewed_audio(db, dry_run=False)
        finally:
            db.close()
        for event_id in ("event_000003", "event_000004"):
            assert os.path.exists(paths[event_id]), (
                f"{event_id} was judged by a human and must keep its audio")

    def test_annotations_survive(self, world):
        db_path, root, _paths = world
        db = EventDatabase(db_path)
        try:
            clear_unreviewed_audio(db, dry_run=False)
            rejected = db.get_event("event_000004")
            assert rejected.decision is Decision.REJECTED
            assert rejected.annotations
        finally:
            db.close()

    def test_metadata_survives(self, world):
        db_path, root, _paths = world
        db = EventDatabase(db_path)
        try:
            clear_unreviewed_audio(db, dry_run=False)
            stored = db.get_event("event_000000")
            assert stored.metadata["classification"]["label"] == "probe"
            assert stored.segmentation_reason is not None
        finally:
            db.close()

    def test_a_file_already_missing_is_reported_not_fatal(self, world):
        db_path, root, paths = world
        os.remove(paths["event_000001"])
        db = EventDatabase(db_path)
        try:
            report = clear_unreviewed_audio(db, dry_run=False)
            assert report.with_audio == 3
            assert "event_000001" in report.missing_files
            # The row is corrected even though there was no file to remove.
            stored = db.get_event("event_000001")
            assert stored.audio_available is False
            # And the others were still cleared.
            assert not os.path.exists(paths["event_000000"])
        finally:
            db.close()

    def test_running_twice_is_harmless(self, world):
        db_path, root, _paths = world
        db = EventDatabase(db_path)
        try:
            first = clear_unreviewed_audio(db, dry_run=False)
            second = clear_unreviewed_audio(db, dry_run=False)
        finally:
            db.close()
        assert first.with_audio == 3
        assert second.would_change_anything is False

    def test_the_retention_snapshot_is_kept_honest(self, world):
        """When a manager is available, the accounting is decremented too."""
        from app.events.retention import RetentionManager

        db_path, root, _paths = world
        db = EventDatabase(db_path)
        try:
            manager = RetentionManager(
                AppConfig().audio_retention, database=db, root=root
            )
            manager.load_accounting()
            manager.reconcile(enforce=False)
            before = manager.accounting.total_bytes
            assert before > 0
            report = clear_unreviewed_audio(db, manager, dry_run=False)
            after = manager.accounting.total_bytes
        finally:
            db.close()
        assert report.freed_bytes > 0
        assert after == before - report.freed_bytes
        assert after > 0, "the judged events' audio is still on disk"

    def test_separation_output_is_kept(self, world):
        """A separation the user paid minutes of CPU for is not deleted."""
        db_path, root, paths = world
        attempt = os.path.join(
            os.path.dirname(paths["event_000000"]), "separation_001"
        )
        os.makedirs(attempt, exist_ok=True)
        isolated = os.path.join(attempt, "isolated.wav")
        with open(isolated, "wb") as handle:
            handle.write(b"\0" * 4096)

        db = EventDatabase(db_path)
        try:
            report = clear_unreviewed_audio(db, dry_run=False)
        finally:
            db.close()
        assert report.retained_separation_bytes == 4096
        assert os.path.exists(isolated), "separation output was deleted"
        # And the report mentions it, so the choice is visible.
        assert "separation" in report.summary()
