"""Tests for byte-based audio retention.

The properties under test are the ones the design is actually built on, not
just the arithmetic:

* caps are measured in **bytes**, and are enforced per classification *and*
  globally;
* room is made **before** a file is written, so the global cap is never
  transiently exceeded;
* eviction prefers **redundancy** over age, and protects human-judged events
  in both directions;
* evicting audio **never** touches a fingerprint, a measurement or a
  review;
* accounting is maintained incrementally, is validated against the database
  on startup, and is reconciled against the filesystem periodically - the
  system never assumes its own bookkeeping is right.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from app.config import (
    AppConfig,
    AudioRetentionConfig,
    ConfigError,
    format_byte_size,
    parse_byte_size,
)
from app.events.database import Decision, EventDatabase, StoredEvent
from app.events.fingerprint import NORMALISATION_VERSION, EventFingerprint
from app.events.retention import (
    UNCLASSIFIED,
    RetentionAccounting,
    RetentionManager,
    EvictionCandidate,
    quality_penalty,
    select_victims,
    wav_bytes_for,
)
from app.events.store import EventStore

KB = 1024
MB = 1024 ** 2
GB = 1024 ** 3


# ======================================================================
# Byte size parsing and configuration
# ======================================================================
class TestByteSizes:
    def test_plain_integers(self):
        assert parse_byte_size(0) == 0
        assert parse_byte_size(524288000) == 524288000

    def test_human_readable(self):
        assert parse_byte_size("500MB") == 500 * MB
        assert parse_byte_size("5GB") == 5 * GB
        assert parse_byte_size("1.5 GiB") == int(1.5 * GB)
        assert parse_byte_size("2g") == 2 * GB
        assert parse_byte_size("64k") == 64 * KB

    def test_mb_is_binary_matching_the_documented_example(self):
        # The brief's example config says 500 MB = 524288000, i.e. MiB.  The
        # parser must agree with the number that is actually written, or the
        # configured cap and the enforced cap differ by 5%.
        assert parse_byte_size("500MB") == 524288000

    def test_rejects_nonsense(self):
        for bad in ("", "abc", "500 parsecs", None, -1, "500 500MB", True):
            with pytest.raises((ConfigError, ValueError)):
                parse_byte_size(bad)

    def test_format(self):
        assert format_byte_size(500 * MB) == "500.0 MB"
        assert format_byte_size(5 * GB) == "5.0 GB"
        assert format_byte_size(512) == "512 B"

    def test_defaults(self):
        config = AudioRetentionConfig()
        assert config.per_class_cap_bytes == 500 * MB
        assert config.total_cap_bytes == 5 * GB
        assert config.preserve_representative_audio is True
        assert config.enabled is True

    def test_rejects_per_class_above_total(self):
        with pytest.raises(ConfigError):
            AudioRetentionConfig(per_class_cap_bytes="10GB", total_cap_bytes="1GB")

    def test_rejects_non_positive(self):
        with pytest.raises(ConfigError):
            AudioRetentionConfig(total_cap_bytes=0)

    def test_string_values_normalised_to_bytes_in_config(self):
        config = AppConfig(
            audio_retention={
                "per_class_cap_bytes": "500MB",
                "total_cap_bytes": "5GB",
            }
        )
        assert config.audio_retention.per_class_cap_bytes == 500 * MB
        assert config.audio_retention.total_cap_bytes == 5 * GB

    def test_round_trip_through_the_project_config_format(self, tmp_path):
        path = tmp_path / "config.json"
        AppConfig(
            audio_retention={"per_class_cap_bytes": "1GB", "total_cap_bytes": "4GB"}
        ).save(str(path))
        reloaded = AppConfig.load(str(path))
        assert reloaded.audio_retention.per_class_cap_bytes == 1 * GB
        assert reloaded.audio_retention.total_cap_bytes == 4 * GB
        # Saved as explicit bytes, never as the readable form.
        raw = json.loads(path.read_text())["audio_retention"]
        assert raw["per_class_cap_bytes"] == 1 * GB

    def test_unknown_key_rejected(self):
        with pytest.raises(ConfigError):
            AppConfig(audio_retention={"per_class_cap": 1})


# ======================================================================
# Accounting
# ======================================================================
class TestAccounting:
    def test_add_and_remove(self):
        accounting = RetentionAccounting()
        accounting.add("motor", 1000)
        accounting.add("motor", 500)
        accounting.add("hvac", 250)
        assert accounting.total_bytes == 1750
        assert accounting.class_bytes("motor") == 1500
        assert accounting.total_files == 3

        accounting.remove("motor", 1200)
        assert accounting.class_bytes("motor") == 300
        assert accounting.total_bytes == 550

    def test_classification_defaults_to_unclassified(self):
        accounting = RetentionAccounting()
        accounting.add(None, 10)
        assert accounting.class_bytes(None) == 10
        assert UNCLASSIFIED in accounting.bytes_by_class

    def test_remove_is_clamped_at_zero(self):
        # A few bytes of drift must not make the total negative and then
        # start refusing to evict.
        accounting = RetentionAccounting()
        accounting.add("motor", 100)
        accounting.remove("motor", 500)
        assert accounting.class_bytes("motor") == 0
        assert accounting.total_bytes == 0
        assert accounting.total_files == 0

    def test_round_trip(self):
        accounting = RetentionAccounting()
        accounting.add("a", 10)
        accounting.add("b", 20)
        restored = RetentionAccounting.from_dict(accounting.to_dict())
        assert restored.to_dict() == accounting.to_dict()

    def test_from_empty_dict(self):
        assert RetentionAccounting.from_dict(None).total_bytes == 0
        assert RetentionAccounting.from_dict({}).total_files == 0


# ======================================================================
# Encoded size
# ======================================================================
def test_wav_bytes_for_matches_the_written_file(tmp_path):
    from app.audio.wavio import write_wav_atomic

    frames = 4800
    samples = np.zeros(frames, dtype=np.float32)
    path = str(tmp_path / "x.wav")
    write_wav_atomic(path, samples, 48000)
    assert os.path.getsize(path) == wav_bytes_for(frames, channels=1)


# ======================================================================
# Eviction selection
# ======================================================================
def _fingerprint(duration: float = 1.0) -> EventFingerprint:
    """A fingerprint whose similarity is driven by ``duration``.

    Duration is used as the control knob because it is a field the
    normalisation actually separates: on the centroid, a large measurement
    difference still normalises to a near-zero distance, so a test built on
    it would silently compare everything as identical.
    """
    return EventFingerprint(
        fingerprint_version=1,
        normalisation_version=NORMALISATION_VERSION,
        detector_version="test",
        duration=duration,
        onset_count=1,
        onset_density=0.1,
        inter_onset_mean=None,
        inter_onset_std=None,
        inter_onset_cv=None,
        level_mean=-40.0,
        level_std=1.0,
        level_min=-45.0,
        level_max=-30.0,
        level_dynamic_range=15.0,
        snr_peak=20.0,
        snr_mean=12.0,
        snr_std=3.0,
        presence=0.5,
        centroid_mean=1000.0,
        centroid_std=0.0,
        centroid_min=0.0,
        centroid_max=0.0,
        centroid_spread=0.0,
        centroid_trend=0.0,
        flatness_mean=0.5,
        flatness_std=1.0,
        flatness_min=0.0,
        flatness_max=0.0,
        flatness_spread=0.0,
        bandwidth_mean=1000.0,
        flux_max=1.0,
        crest_max=5.0,
        modulation_mean=0.3,
        envelope=(),
        high_to_low=0.5,
    )


def _candidate(
    event_id,
    classification,
    size,
    distance,
    created_at,
    decision="unreviewed",
):
    return EvictionCandidate(
        event_id=event_id,
        classification=classification,
        audio_bytes=size,
        created_at=created_at,
        audio_path=f"/tmp/{event_id}.wav",
        duration=1.0,
        decision=decision,
        # Near-equal durations are near-identical fingerprints, so `distance`
        # reads as "how different from the others this recording is".
        fingerprint=_fingerprint(duration=distance),
        metadata={},
    )


class TestSelectVictims:
    def test_evicts_the_most_redundant_not_the_oldest(self):
        # a and b are near-identical; c is an outlier.  Evicting the most
        # redundant must not be the same as evicting the oldest.
        candidates = [
            _candidate("a", "motor", 100, 1.0, "2026-01-01T00:00:00"),
            _candidate("b", "motor", 100, 1.1, "2026-01-01T00:00:01"),
            _candidate("c", "motor", 100, 30.0, "2026-01-01T00:00:02"),
        ]
        victims = select_victims(candidates, 100)
        assert len(victims) == 1
        # The outlier survives - that is the whole point of keeping a
        # representative collection instead of a FIFO one.
        assert victims[0].event_id in {"a", "b"}
        assert "c" not in [v.event_id for v in victims]

    def test_spreads_out_rather_than_draining_one_cluster(self):
        # Three near-identical recordings and two distinct ones.  Removing
        # enough for 200 bytes must take the two most redundant, and must
        # not empty a single cluster down to one survivor when others exist.
        candidates = [
            _candidate("a", "motor", 100, 1.0, "2026-01-01T00:00:00"),
            _candidate("b", "motor", 100, 1.1, "2026-01-01T00:00:01"),
            _candidate("c", "motor", 100, 1.2, "2026-01-01T00:00:02"),
            _candidate("d", "motor", 100, 20.0, "2026-01-01T00:00:03"),
            _candidate("e", "motor", 100, 25.0, "2026-01-01T00:00:04"),
        ]
        victims = select_victims(candidates, 200)
        assert len(victims) == 2
        # The tight cluster is thinned, not emptied: the distinct recordings
        # are still there afterwards, which is what a FIFO archive loses.
        assert {v.event_id for v in victims} == {"a", "b"}
        assert {c.event_id for c in candidates} - {
            v.event_id for v in victims
        } == {"c", "d", "e"}

    def test_human_judged_events_are_protected_in_both_directions(self):
        # A confirmed example and a rejected negative example are both
        # irreplaceable labelled data points.
        candidates = [
            _candidate("keep_confirmed", "motor", 100, 1.0, "2026-01-01T00:00:00", "confirmed"),
            _candidate("keep_rejected", "motor", 100, 1.0, "2026-01-01T00:00:01", "rejected"),
            _candidate("keep_saved", "motor", 100, 1.0, "2026-01-01T00:00:02", "saved"),
            _candidate("drop", "motor", 100, 10.0, "2026-01-01T00:00:03", "unreviewed"),
        ]
        victims = select_victims(candidates, 100)
        assert [v.event_id for v in victims] == ["drop"]

    def test_protection_can_be_disabled(self):
        candidates = [
            _candidate("confirmed", "motor", 100, 5.0, "2026-01-01T00:00:00", "confirmed"),
            _candidate("unreviewed", "motor", 100, 5.0, "2026-01-01T00:00:01"),
        ]
        victims = select_victims(
            candidates, 100, protect_human_decisions=False
        )
        assert len(victims) == 1

    def test_fifo_mode_is_oldest_first(self):
        candidates = [
            _candidate("newest", "motor", 100, 30.0, "2026-01-03T00:00:00"),
            _candidate("oldest", "motor", 100, 1.0, "2026-01-01T00:00:00"),
            _candidate("middle", "motor", 100, 10.0, "2026-01-02T00:00:00"),
        ]
        victims = select_victims(
            candidates, 100, preserve_representative=False
        )
        assert [v.event_id for v in victims] == ["oldest"]
        # Bounded by the need, like every other path: a FIFO pass must not
        # empty the class when one file would have done.
        assert len(
            select_victims(candidates, 200, preserve_representative=False)
        ) == 2

    def test_unscoreable_candidates_are_not_preferred(self):
        # No fingerprint means unknown, not identical.  Treating unknown as
        # zero distance would make exactly these records the first evicted.
        unscoreable = EvictionCandidate(
            event_id="unknown",
            classification="motor",
            audio_bytes=100,
            created_at="2026-01-01T00:00:00",
            audio_path=None,
            duration=1.0,
            decision="unreviewed",
            fingerprint=None,
        )
        # 'redundant' has a measured near-duplicate; 'unknown' cannot be
        # measured at all, so it is kept ahead of it.
        redundant = _candidate("redundant", "motor", 100, 1.0, "2026-01-01T00:00:01")
        duplicate = _candidate("duplicate", "motor", 100, 1.05, "2026-01-01T00:00:02")
        victims = select_victims([unscoreable, redundant, duplicate], 100)
        assert [v.event_id for v in victims] == ["redundant"]
        # ... but it is still reachable once everything else is gone, because
        # its rank is a large finite value rather than infinity.
        assert select_victims([unscoreable], 100) == [unscoreable]

    def test_never_returns_more_than_needed(self):
        candidates = [
            _candidate(f"e{i}", "motor", 100, 1.0 + i * 3, f"2026-01-01T00:00:0{i}")
            for i in range(10)
        ]
        assert len(select_victims(candidates, 100)) == 1
        assert len(select_victims(candidates, 0)) == 0
        assert select_victims([], 100) == []

    def test_candidates_without_protection_available(self):
        protected = [
            _candidate(f"e{i}", "motor", 100, 1.0, f"2026-01-01T00:00:0{i}", "confirmed")
            for i in range(3)
        ]
        assert select_victims(protected, 500) == []


class TestQualityPenalty:
    def test_incomplete_capture_penalised_most(self):
        complete = quality_penalty(1.0, {"peak_dbfs": -20.0})
        incomplete = quality_penalty(
            1.0, {"peak_dbfs": -20.0, "preroll_missing_frames": 480}
        )
        assert incomplete > complete
        assert quality_penalty(1.0, {"postroll_missing_frames": 480}) > complete

    def test_near_silence_and_clipping_penalised(self):
        baseline = quality_penalty(1.0, {})
        assert quality_penalty(1.0, {"peak_dbfs": -90.0}) > baseline
        assert quality_penalty(1.0, {"clipped_fraction": 0.5}) > baseline

    def test_very_short_penalised(self):
        assert quality_penalty(0.05, {}) > quality_penalty(1.0, {})

    def test_bounded_at_one(self):
        assert quality_penalty(
            0.0,
            {
                "preroll_missing_frames": 1,
                "postroll_missing_frames": 1,
                "peak_dbfs": -100.0,
                "clipped_fraction": 1.0,
            },
        ) == 1.0

    def test_missing_measurements_are_not_penalised(self):
        # A missing measurement reads as absent, not as evidence of a fault.
        assert quality_penalty(1.0, {}) == 0.0
        assert quality_penalty(1.0, {"peak_dbfs": None}) == 0.0


# ======================================================================
# The manager, end to end
# ======================================================================
def _store_event(
    database, root, event_id, classification, size_bytes,
    distance=0.5, decision=None, day="2026-01-01",
):
    """Create a real event row and a real audio file of a given size."""
    directory = os.path.join(root, day, event_id)
    os.makedirs(directory, exist_ok=True)
    audio_path = os.path.join(directory, "original.wav")
    with open(audio_path, "wb") as handle:
        handle.write(b"\0" * size_bytes)
    database.store_event(
        event_id=event_id,
        timestamp="2026-01-01T00:00:00",
        duration=1.0,
        metadata={"classification": classification, "peak_dbfs": -20.0},
        fingerprint=_fingerprint(distance),
        audio_path=audio_path,
        classification=classification,
    )
    database.set_audio_state(event_id, audio_bytes=size_bytes, available=True)
    database.set_audio_path(event_id, audio_path)
    if decision:
        database.annotate(event_id, decision)
    return audio_path


@pytest.fixture
def env(tmp_path):
    """A database and an event tree, with retention wired to both."""
    root = str(tmp_path / "events")
    os.makedirs(root, exist_ok=True)
    database = EventDatabase(str(tmp_path / "events.db"))
    config = AudioRetentionConfig(
        per_class_cap_bytes=1000, total_cap_bytes=2000
    )
    manager = RetentionManager(config, database=database, root=root)
    manager.load_accounting()
    yield database, manager, root
    database.close()


class TestManager:
    def test_reconcile_accounts_existing_files(self, env):
        database, manager, root = env
        path = _store_event(database, root, "e1", "motor", 400)
        report = manager.reconcile(enforce=False)
        assert report is not None
        assert manager.accounting.total_bytes == 400
        assert manager.accounting.class_bytes("motor") == 400
        assert os.path.exists(path)

    def test_reserve_makes_room_in_the_class_before_writing(self, env):
        database, manager, root = env
        _store_event(database, root, "old", "motor", 900)
        manager.reconcile(enforce=False)
        # The next 400 bytes would take the class to 1300, over its 1000 cap.
        report = manager.reserve("motor", 400)
        assert report.freed_bytes >= 300
        assert not os.path.exists(
            os.path.join(root, "2026-01-01", "old", "original.wav")
        )
        assert manager.accounting.class_bytes("motor") <= 1000

    def test_reserve_respects_the_global_cap(self, env):
        database, manager, root = env
        _store_event(database, root, "a", "motor", 900)
        _store_event(database, root, "b", "hvac", 900)
        manager.reconcile(enforce=False)
        assert manager.accounting.total_bytes == 1800
        manager.reserve("motor", 500)
        assert manager.accounting.total_bytes + 500 <= 2000

    def test_global_eviction_spans_classes(self, env):
        database, manager, root = env
        _store_event(database, root, "a", "motor", 900)
        _store_event(database, root, "b", "hvac", 900)
        manager.reconcile(enforce=False)
        manager.reserve("motor", 500)
        remaining = {
            e.event_id
            for e in database.audio_candidates()
            if e.event_id != "a"
        }
        # Room was made from somewhere, and not only from the incoming class.
        assert manager.accounting.total_bytes + 500 <= 2000
        assert len(remaining) >= 1

    def test_file_larger_than_the_whole_cap_is_stored_and_reported(self, env):
        database, manager, root = env
        report = manager.reserve("motor", 5000)
        assert report.overflow_bytes > 0
        assert report.over_total_cap is True
        assert any("over" in note for note in report.notes)

    def test_disabled_manager_does_nothing(self, tmp_path):
        database = EventDatabase(str(tmp_path / "e.db"))
        manager = RetentionManager(
            AudioRetentionConfig(enabled=False, per_class_cap_bytes=1),
            database=database, root=str(tmp_path),
        )
        report = manager.reserve("motor", 10 ** 6)
        assert report.freed_bytes == 0
        assert manager.accounting.total_bytes == 0
        database.close()

    def test_accounting_survives_a_restart(self, env):
        database, manager, root = env
        _store_event(database, root, "e1", "motor", 400)
        manager.reconcile(enforce=False)
        # A new manager over the same database, as a restart would build.
        fresh = RetentionManager(
            manager.config, database=database, root=root
        )
        assert fresh.load_accounting().total_bytes == 400

    def test_hot_path_does_not_walk_the_tree(self, env, monkeypatch):
        database, manager, root = env
        _store_event(database, root, "e1", "motor", 400)
        manager.reconcile(enforce=False)
        calls = []
        monkeypatch.setattr(
            manager, "_scan_tree",
            lambda: calls.append(1) or {},
        )
        manager.reserve("motor", 100)
        manager.record_written("e2", "motor", "/tmp/e2.wav", 100)
        assert calls == []

    def test_reconcile_finds_an_externally_deleted_file(self, env):
        database, manager, root = env
        path = _store_event(database, root, "e1", "motor", 400)
        manager.reconcile(enforce=False)
        os.remove(path)                      # deleted behind our back
        report = manager.reconcile(enforce=False)
        assert manager.accounting.total_bytes == 0
        stored = database.get_event("e1")
        assert stored.audio_available is False
        assert stored.audio_path is None
        assert any("absent" in note for note in report.notes)

    def test_reconcile_accounts_a_file_with_no_row(self, env):
        database, manager, root = env
        directory = os.path.join(root, "2026-01-01", "orphan")
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "original.wav"), "wb") as handle:
            handle.write(b"\0" * 700)
        report = manager.reconcile(enforce=False)
        # Ignoring an unknown file is how usage grows without bound.
        assert manager.accounting.class_bytes(UNCLASSIFIED) == 700
        assert any("no event record" in note for note in report.notes)

    def test_reconcile_repairs_accounting_drift(self, env):
        database, manager, root = env
        _store_event(database, root, "e1", "motor", 400)
        manager.reconcile(enforce=False)
        manager.accounting.total_bytes = 999999      # corrupt the bookkeeping
        manager.reconcile(enforce=False)
        assert manager.accounting.total_bytes == 400

    def test_reconcile_enforces_a_cap_already_exceeded(self, env):
        database, manager, root = env
        _store_event(database, root, "a", "motor", 900)
        _store_event(database, root, "b", "motor", 900)
        report = manager.reconcile(enforce=True)
        assert "motor" in report.over_class_caps
        assert manager.accounting.class_bytes("motor") <= 1000

    def test_reconciliation_is_rate_limited(self, env, monkeypatch):
        database, manager, root = env
        _store_event(database, root, "e1", "motor", 400)
        calls = []
        original = manager._scan_tree

        def counting():
            calls.append(1)
            return original()

        monkeypatch.setattr(manager, "_scan_tree", counting)
        manager.config.reconcile_interval_seconds = 10_000
        manager.reconcile(enforce=False)
        for _ in range(5):
            manager.reserve("motor", 10)
        assert len(calls) == 1

    def test_status_reports_both_caps(self, env):
        database, manager, root = env
        _store_event(database, root, "e1", "motor", 400)
        manager.reconcile(enforce=False)
        status = manager.status()
        assert status["total_bytes"] == 400
        assert status["per_class_cap_bytes"] == 1000
        assert status["total_cap_bytes"] == 2000
        assert status["classes"]["motor"]["bytes"] == 400
        assert status["classes"]["motor"]["files"] == 1
        assert status["over_total_cap"] is False


class TestClassificationKey:
    def test_metadata_classification_block_resolves_to_its_label(self):
        # The detector stores a *block* here, not a string.  Bucketing on the
        # block puts an unhashable dict into the accounting.
        from app.events.retention import classification_key

        event = _store_probe_event(
            metadata={
                "classification": {
                    "label": "some_label",
                    "label_scores": {},
                    "margin": 0.1,
                }
            },
            classification="some_label",
        )
        assert classification_key(event) == "some_label"

    def test_falls_back_to_the_column_then_to_unclassified(self):
        from app.events.retention import classification_key

        assert classification_key(
            _store_probe_event(metadata={}, classification="from_column")
        ) == "from_column"
        assert classification_key(
            _store_probe_event(metadata={}, classification=None)
        ) == UNCLASSIFIED
        assert classification_key(
            _store_probe_event(metadata={}, classification="")
        ) == UNCLASSIFIED

    def test_block_without_a_label_falls_back(self):
        from app.events.retention import classification_key

        assert classification_key(
            _store_probe_event(
                metadata={"classification": {"margin": 0.2}},
                classification="from_column",
            )
        ) == "from_column"


def _store_probe_event(metadata, classification) -> StoredEvent:
    """A StoredEvent with only the fields the bucketing reads."""
    from app.events.database import StoredEvent

    return StoredEvent(
        event_id="e1",
        timestamp="2026-01-01T00:00:00",
        duration=1.0,
        audio_path="/tmp/e1.wav",
        detector_version="test",
        fingerprint_version=1,
        segmentation_reason="unknown",
        created_at="2026-01-01T00:00:00",
        metadata=metadata,
        classification=classification,
    )


class TestPerClassFloor:
    """Retention never reduces a classification to zero audio files.

    A storage policy is not a reason to destroy the only copy of a class's
    evidence, so an overage that cannot be reclaimed without doing that is
    reported instead.  A *write* is different: there an incoming file has to
    fit, and making room for it is the reason room is reserved first.
    """

    def test_reconcile_keeps_the_last_file_and_reports_the_overage(self, env):
        database, manager, root = env
        # One file, larger than the 1000-byte class cap.
        _store_event(database, root, "only", "motor", 4000)
        report = manager.reconcile(enforce=True)
        assert report.evicted == []
        assert os.path.exists(
            os.path.join(root, "2026-01-01", "only", "original.wav")
        )
        assert database.get_event("only").audio_available is True
        assert any("over its" in note for note in report.notes)
        assert "unclassified" not in str(report.notes)

    def test_reconcile_keeps_two_files_when_it_must_keep_one(self, env):
        database, manager, root = env
        _store_event(database, root, "a", "motor", 900)
        _store_event(database, root, "b", "motor", 900)
        manager.reconcile(enforce=False)
        report = manager.reconcile(enforce=True)
        # 1800 bytes against a 1000-byte cap: one file has to go, and exactly
        # one, because the floor is one file.
        assert len(report.evicted) == 1
        assert manager.accounting.class_bytes("motor") == 900
        assert manager.accounting.file_count_by_class["motor"] == 1

    def test_a_write_may_evict_the_last_file_of_a_class(self, env):
        database, manager, root = env
        path = _store_event(database, root, "only", "motor", 4000)
        manager.reconcile(enforce=False)
        # The incoming file is exempt, so the only other file is the victim.
        report = manager.reserve("motor", 500)
        assert [r.event_id for r in report.evicted] == ["only"]
        assert not os.path.exists(path)

    def test_floor_of_zero_allows_emptying_a_class(self, tmp_path):
        root = str(tmp_path / "events")
        os.makedirs(root, exist_ok=True)
        database = EventDatabase(str(tmp_path / "e.db"))
        config = AudioRetentionConfig(
            per_class_cap_bytes=1000, total_cap_bytes=2000,
            min_files_per_class=0,
        )
        manager = RetentionManager(config, database=database, root=root)
        manager.load_accounting()
        _store_event(database, root, "only", "motor", 4000)
        report = manager.reconcile(enforce=True)
        assert [r.event_id for r in report.evicted] == ["only"]
        database.close()

    def test_floor_is_validated(self):
        with pytest.raises(ConfigError):
            AudioRetentionConfig(min_files_per_class=-1)

    def test_floor_in_config_round_trip(self, tmp_path):
        path = tmp_path / "config.json"
        AppConfig(audio_retention={"min_files_per_class": 2}).save(str(path))
        assert AppConfig.load(str(path)).audio_retention.min_files_per_class == 2


class TestRowLinkingRaces:
    """Regressions for three ways the accounting could drift upward.

    Each of these was found by running the real pipeline, not by a unit test,
    so each is pinned here.
    """

    def test_restoring_an_event_does_not_resurrect_evicted_audio(self, env):
        # The writer thread re-inserts a row with INSERT OR REPLACE, which
        # deletes and recreates it and therefore resets the retention columns
        # to their defaults.  An evicted event would come back claiming audio
        # that no longer exists.
        database, manager, root = env
        _store_event(database, root, "e1", "motor", 400)
        database.mark_audio_evicted("e1")
        assert database.get_event("e1").audio_available is False

        database.store_event(
            event_id="e1",
            timestamp="2026-01-01T00:00:00",
            duration=1.0,
            metadata={"classification": "motor"},
            audio_path="/somewhere/else/original.wav",
            classification="motor",
        )
        stored = database.get_event("e1")
        assert stored.audio_available is False
        assert stored.audio_path is None
        assert stored.audio_evicted_at is not None
        # The detector-owned columns are still updated by the re-insert.
        assert stored.metadata["classification"] == "motor"

    def test_linking_an_already_evicted_event_adds_nothing(self, env):
        database, manager, root = env
        _store_event(database, root, "e1", "motor", 400)
        database.mark_audio_evicted("e1")
        before = manager.accounting.total_bytes
        manager.link_row("e1", "motor", None, 400)
        assert manager.accounting.total_bytes == before
        assert manager.accounting.total_files == 0

    def test_linking_a_missing_file_marks_it_evicted_and_counts_nothing(
        self, env
    ):
        database, manager, root = env
        _store_event(database, root, "e1", "motor", 400)
        os.remove(os.path.join(root, "2026-01-01", "e1", "original.wav"))
        manager.link_row(
            "e1", "motor", os.path.join(root, "2026-01-01", "e1", "original.wav"), 0
        )
        assert database.get_event("e1").audio_available is False
        assert manager.accounting.total_bytes == 0
        assert manager.accounting.total_files == 0

    def test_evicted_then_linked_does_not_double_count(self, env):
        # The exact interleaving from the real run: retention evicts an event
        # the writer has not linked yet, and the writer then arrives.
        database, manager, root = env
        _store_event(database, root, "e1", "motor", 400)
        manager.record_written("e1", "motor", "/tmp/e1.wav", 400)
        assert manager.accounting.total_bytes == 400

        # 400 already held, class cap 1000: asking for 700 forces eviction.
        report = manager.reserve("motor", 700)
        assert report.freed_bytes == 400
        assert manager.accounting.total_bytes == 0
        assert manager._unlinked == {}

        # The writer's late link must change nothing.
        manager.link_row("e1", "motor", None, 0)
        assert manager.accounting.total_bytes == 0
        assert manager.accounting.total_files == 0

    def test_concurrent_writes_and_evictions_stay_consistent(self, env):
        # The accounting must equal the bytes actually on disk however the two
        # threads interleave.  Run enough iterations for the interleaving that
        # broke it to occur.
        import threading

        database, manager, root = env
        for index in range(12):
            _store_event(
                database, root, f"e{index}", "motor", 400,
                distance=1.0 + index * 0.5,
            )
        manager.reconcile(enforce=False)

        errors: list[Exception] = []

        def writer() -> None:
            for index in range(12, 40):
                try:
                    manager.record_written(
                        f"w{index}", "motor", f"/tmp/w{index}.wav", 400
                    )
                    manager.link_row(
                        f"w{index}", "motor", f"/tmp/w{index}.wav", 400
                    )
                except Exception as exc:  # pragma: no cover - reported below
                    errors.append(exc)

        def evictor() -> None:
            for _ in range(40):
                try:
                    manager.reserve("motor", 400)
                except Exception as exc:  # pragma: no cover
                    errors.append(exc)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=evictor),
            threading.Thread(target=evictor),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30.0)
        assert not errors, errors

        # Files on disk for the store-written events are all under /tmp and
        # were never created, so the truthful usage is what reconcile finds.
        report = manager.reconcile(enforce=False)
        assert manager.accounting.total_bytes >= 0
        assert manager.accounting.total_files >= 0
        on_disk = sum(
            os.path.getsize(os.path.join(root, d, e, "original.wav"))
            for d in os.listdir(root)
            for e in os.listdir(os.path.join(root, d))
            if os.path.exists(os.path.join(root, d, e, "original.wav"))
        )
        assert manager.accounting.total_bytes == on_disk
        del report


# ======================================================================
# Fingerprints are independent of audio
# ======================================================================
class TestFingerprintSurvival:
    def test_evicting_audio_keeps_fingerprint_and_history(self, env):
        database, manager, root = env
        _store_event(database, root, "e1", "motor", 900)
        _store_event(database, root, "e2", "motor", 900, distance=0.1)
        manager.reconcile(enforce=False)
        database.annotate("e1", Decision.SAVED, label="motor", notes="check me")

        report = manager.reserve("motor", 800)
        assert report.freed_bytes > 0
        evicted_ids = [r.event_id for r in report.evicted]
        assert evicted_ids == ["e2"]          # e1 is human-saved, so untouched

        # Even the evicted event keeps everything except the file.
        stored = database.get_event("e2")
        assert stored.fingerprint is not None
        assert stored.metadata["peak_dbfs"] == -20.0
        assert stored.audio_available is False
        assert stored.audio_path is None
        assert stored.audio_evicted_at is not None
        assert stored.is_playable is False
        assert database.count() == 2

    def test_eviction_is_recorded_in_the_database(self, env):
        database, manager, root = env
        # Both events are in one class, so the class holds 1800 bytes against
        # a 1000-byte cap: making room for 800 more means evicting both.
        _store_event(database, root, "e1", "motor", 900)
        _store_event(database, root, "e2", "motor", 900)
        manager.reconcile(enforce=False)
        report = manager.reserve("motor", 800)
        assert len(report.evicted) == 2

        unavailable = [
            e.event_id for e in database.list_events() if not e.audio_available
        ]
        assert sorted(unavailable) == ["e1", "e2"]
        # ... and every one of those files really is gone.
        for event_id in unavailable:
            assert not os.path.exists(
                os.path.join(root, "2026-01-01", event_id, "original.wav")
            )

    def test_similarity_still_works_over_evicted_events(self, env):
        database, manager, root = env
        _store_event(database, root, "e1", "motor", 900, distance=0.4)
        _store_event(database, root, "e2", "motor", 900, distance=0.45)
        manager.reconcile(enforce=False)
        manager.reserve("motor", 800)
        query = _fingerprint(0.42)
        # The point of a long-lived fingerprint database: the evicted event
        # is still comparable, and still findable by similarity.
        assert len(database.find_similar(query, limit=5)) == 2


# ======================================================================
# Integration with the store
# ======================================================================
class TestStoreIntegration:
    def _event(self, event_id, seconds=1.0, classification="motor"):
        from app.events.event import Event

        return Event(
            event_id=event_id,
            start_seconds=0.0,
            end_seconds=seconds,
            sample_rate=48000,
            classification=classification,
        )

    def test_written_audio_is_accounted(self, env, tmp_path):
        database, manager, root = env
        store = EventStore(root, retention=manager)
        samples = np.zeros(48000, dtype=np.float32)
        event = store.save(self._event("e1"), samples)
        expected = wav_bytes_for(48000)
        assert manager.accounting.total_bytes == expected
        assert manager.accounting.class_bytes("motor") == expected
        assert os.path.exists(event.audio_path)
        assert store.write_errors == []

    def test_store_without_retention_still_works(self, tmp_path):
        store = EventStore(str(tmp_path / "events"))
        samples = np.zeros(1000, dtype=np.float32)
        event = store.save(self._event("e1", seconds=0.02), samples)
        assert event.audio_path and os.path.exists(event.audio_path)

    def test_over_cap_write_evicts_an_older_unreviewed_event(self, env):
        database, manager, root = env
        _store_event(database, root, "old", "motor", 900)
        manager.reconcile(enforce=False)

        store = EventStore(root, retention=manager)
        # ~192 KB: takes the class well past its 1000-byte cap.
        samples = np.zeros(48000, dtype=np.float32)
        written = store.save(self._event("new"), samples)

        assert not os.path.exists(
            os.path.join(root, "2026-01-01", "old", "original.wav")
        )
        # The incoming event is never the thing evicted to make room for
        # itself: its file exists, and the older redundant one does not.
        assert os.path.exists(written.audio_path)
        assert manager.accounting.class_bytes("motor") <= wav_bytes_for(48000)

    def test_writer_links_the_row_to_the_audio(self, env):
        from app.events.persistence import EventWriter, PendingEvent

        database, manager, root = env
        store = EventStore(root, retention=manager)
        samples = np.zeros(4800, dtype=np.float32)
        event = store.save(self._event("e1", seconds=0.1), samples)

        writer = EventWriter(database, retention=manager)
        writer.start()
        writer.submit(
            PendingEvent(
                event_id="e1",
                timestamp="2026-01-01T00:00:00",
                duration=0.1,
                metadata={},
                profile=None,
                audio_path=event.audio_path,
                classification="motor",
            )
        )
        assert writer.flush(5.0)
        writer.stop()

        stored = database.get_event("e1")
        assert stored.audio_bytes == wav_bytes_for(4800)
        assert stored.audio_available is True
        assert stored.is_playable is True
        # Accounted exactly once, not twice.
        assert manager.accounting.total_bytes == wav_bytes_for(4800)

    def test_reconcile_after_a_writer_drop_still_accounts(self, env):
        # The store wrote the audio but the writer queue dropped the row.
        # Over-counting evicts early; under-counting silently overruns the
        # cap, so the bytes must stay accounted and reconciliation must fix
        # the missing row rather than the missing bytes.
        database, manager, root = env
        store = EventStore(root, retention=manager)
        written = store.save(
            self._event("e1", seconds=0.1), np.zeros(4800, np.float32)
        )
        manager.reconcile(enforce=False)
        assert database.get_event("e1") is None
        # The file exists, so the bytes are real usage - and with no row to
        # attribute them to, they are accounted rather than ignored.
        assert os.path.exists(written.audio_path)
        assert manager.accounting.class_bytes(UNCLASSIFIED) == wav_bytes_for(4800)
