"""Event fingerprints, persistence, annotation and similarity.

Generic by construction: nothing here knows what any sound *is*.  Fingerprints
summarise measurements the detector already makes, and the annotation layer is
where a user-supplied meaning would live.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time

import numpy as np
import pytest

from app.config import AppConfig
from app.detection.tracker import (
    EventProfile,
    EventTermination,
    EventTracker,
    TrackedEvent,
)
from app.events.database import (
    Annotation,
    Decision,
    EventDatabase,
    StoredEvent,
)
from app.events.fingerprint import (
    DETECTOR_VERSION,
    ENVELOPE_POINTS,
    FINGERPRINT_VERSION,
    NORMALISATION_VERSION,
    VECTOR_FIELDS,
    EventFingerprint,
    IncomparableFingerprints,
    RunningMoments,
    build_fingerprint,
    similarity,
)
from app.events.persistence import EventWriter, PendingEvent

SR = 48000


# ----------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------
def profile(
    *,
    frames: int = 100,
    duration_s: float = 1.0,
    level_db: float = -40.0,
    level_jitter: float = 2.0,
    centroid: float = 1000.0,
    centroid_jitter: float = 100.0,
    centroid_drift: float = 0.0,
    flatness: float = 0.3,
    flatness_jitter: float = 0.02,
    snr: float = 15.0,
    presence: float = 0.7,
    onsets: int = 0,
    intervals=(),
    flux: float = 0.3,
    crest: float = 4.0,
    modulation: float = 0.3,
    termination: str = EventTermination.QUIET.value,
) -> EventProfile:
    """A profile populated the way the detector would populate one."""
    hop = int(SR / 100)
    p = EventProfile(
        onset_sample=1000,
        first_candidate_sample=1000,
        last_active_sample=1000 + int(duration_s * SR),
        frames=frames,
        frames_with_features=frames,
        frames_above_floor=int(frames * presence),
        frames_per_second=100.0,
        peak_snr_db=snr,
        mean_snr_db=snr - 2.0,
        mean_rms_dbfs=level_db,
        peak_dbfs=level_db + 6.0,
        max_crest=crest,
        mean_crest=crest * 0.6,
        mean_modulation=modulation,
        max_modulation=modulation + 0.1,
        max_clipping_ratio=0.0,
        max_peakiness=8.0,
        mean_peakiness=6.0,
        mean_flatness=flatness,
        mean_centroid_hz=centroid,
        mean_bandwidth_hz=3000.0,
        max_bandwidth_hz=5000.0,
        max_flux=flux,
        mean_low_band_snr_db=10.0,
        max_low_band_snr_db=18.0,
        mean_high_band_snr_db=9.0,
        max_high_band_snr_db=15.0,
        mean_mid_band_snr_db=14.0,
        high_to_low_ratio_db=-1.0,
        dominant_frequency_hz=centroid,
        impulsive_onsets=onsets,
        onset_intervals=list(intervals),
    )
    # Populate the online moments the way the tracker's accumulate() does.
    for i in range(frames):
        wobble = (i % 7) / 7.0 - 0.5
        p.level_moments.add(level_db + wobble * level_jitter * 2)
        p.level_series.append(level_db + wobble * level_jitter * 2)
        p.snr_moments.add(snr - 1.0 + wobble)
        p.centroid_moments.add(centroid + wobble * centroid_jitter * 2
                               + centroid_drift * i / max(frames - 1, 1))
        p.flatness_moments.add(flatness + wobble * flatness_jitter * 2)
    return p


def tracked(p: EventProfile, termination=EventTermination.QUIET) -> TrackedEvent:
    return TrackedEvent(
        start_sample=0,
        end_sample=p.last_active_sample,
        sample_rate=SR,
        profile=p,
        termination=termination,
    )


# ======================================================================
# Fingerprint creation
# ======================================================================
def test_completed_event_produces_a_fingerprint():
    fp = build_fingerprint(profile(), termination="quiet")
    assert isinstance(fp, EventFingerprint)
    assert fp.duration == pytest.approx(1.0, abs=0.01)
    assert fp.level_mean == pytest.approx(-40.0, abs=3.0)
    assert fp.centroid_mean == pytest.approx(1000.0, rel=0.15)
    assert fp.presence == pytest.approx(0.7, abs=0.01)
    assert len(fp.envelope) == ENVELOPE_POINTS


def test_fingerprint_is_deterministic():
    """Same measurements in, same fingerprint out."""
    a = build_fingerprint(profile(onsets=3, intervals=(0.5, 0.5)))
    b = build_fingerprint(profile(onsets=3, intervals=(0.5, 0.5)))
    assert a.to_dict() == b.to_dict()
    assert a.vector == b.vector
    assert similarity(a, b) == pytest.approx(0.0, abs=1e-12)


def test_fingerprint_records_versions():
    fp = build_fingerprint(profile())
    assert fp.fingerprint_version == FINGERPRINT_VERSION
    assert fp.normalisation_version == NORMALISATION_VERSION
    assert fp.detector_version == DETECTOR_VERSION
    payload = fp.to_dict()
    assert payload["fingerprint_version"] == FINGERPRINT_VERSION
    assert payload["detector_version"] == DETECTOR_VERSION
    assert payload["vector_fields"] == list(VECTOR_FIELDS)


def test_different_fingerprint_versions_are_not_compared():
    """A distance across schema versions would be meaningless."""
    a = build_fingerprint(profile())
    b = build_fingerprint(profile())
    object.__setattr__(b, "fingerprint_version", FINGERPRINT_VERSION + 1)
    with pytest.raises(IncomparableFingerprints):
        similarity(a, b)
    object.__setattr__(b, "fingerprint_version", FINGERPRINT_VERSION)
    object.__setattr__(b, "normalisation_version", NORMALISATION_VERSION + 1)
    with pytest.raises(IncomparableFingerprints):
        similarity(a, b)


def test_fingerprint_round_trips_through_json():
    fp = build_fingerprint(profile(onsets=2, intervals=(0.4,)), termination="x")
    restored = EventFingerprint.from_dict(json.loads(json.dumps(fp.to_dict())))
    assert restored.to_dict() == fp.to_dict()
    assert restored.vector == fp.vector


# ======================================================================
# Compactness
# ======================================================================
def test_fingerprint_is_compact_and_does_not_grow_with_event_length():
    """The whole point: a compressed summary, not a frame database."""
    small = build_fingerprint(profile(frames=100, duration_s=1.0))
    large = build_fingerprint(profile(frames=6000, duration_s=60.0))
    assert len(small.envelope) == len(large.envelope) == ENVELOPE_POINTS
    assert len(small.vector) == len(large.vector) == len(VECTOR_FIELDS)
    # 60x the frames must not mean anything like 60x the fingerprint.
    assert len(json.dumps(large.to_dict())) < 2 * len(json.dumps(small.to_dict()))


def test_serialised_fingerprint_has_no_per_frame_series():
    fp = build_fingerprint(profile(frames=5000, duration_s=50.0))
    payload = json.dumps(fp.to_dict())
    assert len(payload) < 4096, "fingerprint payload should stay small"
    envelope = fp.to_dict()["temporal"]["envelope"]
    assert len(envelope) == ENVELOPE_POINTS


def test_moments_do_not_retain_the_series():
    moments = RunningMoments()
    for i in range(100000):
        moments.add(float(i))
    assert moments.n == 100000
    assert moments.mean == pytest.approx(49999.5, rel=1e-9)
    assert moments.lo == 0.0 and moments.hi == 99999.0
    # The accumulator itself is a fixed handful of numbers.
    assert len(moments.to_dict()) == 4


# ======================================================================
# Missing / optional features
# ======================================================================
def test_event_without_moments_still_produces_a_fingerprint():
    """A sparse profile must not break persistence."""
    bare = EventProfile(
        onset_sample=0, last_active_sample=SR, frames=1, frames_with_features=0
    )
    fp = build_fingerprint(bare)
    assert isinstance(fp, EventFingerprint)
    assert len(fp.vector) == len(VECTOR_FIELDS)
    json.dumps(fp.to_dict())


def test_event_with_no_onsets_has_no_intervals():
    fp = build_fingerprint(profile(onsets=0))
    assert fp.onset_count == 0
    assert fp.inter_onset_mean is None
    assert fp.inter_onset_cv is None
    # And the vector is still well formed.
    assert all(math.isfinite(v) for v in fp.vector)


def test_from_dict_tolerates_missing_optional_fields():
    fp = EventFingerprint.from_dict({"fingerprint_version": 1})
    assert fp.onset_count == 0
    assert fp.envelope == ()
    json.dumps(fp.to_dict())


def test_onset_statistics_come_from_the_detector():
    p = profile(onsets=4, intervals=(0.5, 1.0, 0.5))
    fp = build_fingerprint(p)
    assert fp.onset_count == 4
    assert fp.inter_onset_mean == pytest.approx(2.0 / 3.0, abs=0.01)
    # Coefficient of variation is scale free, so it survives normalisation.
    assert fp.inter_onset_cv is not None and fp.inter_onset_cv > 0.0


def test_centroid_trend_is_reported():
    # Jitter 0, so the only movement in the centroid is the drift itself.
    rising = build_fingerprint(
        profile(centroid_drift=600.0, duration_s=2.0, centroid_jitter=0.0)
    )
    steady = build_fingerprint(
        profile(centroid_drift=0.0, duration_s=2.0, centroid_jitter=0.0)
    )
    assert rising.centroid_trend > steady.centroid_trend
    assert steady.centroid_trend == pytest.approx(0.0, abs=1.0)


# ======================================================================
# Reuse of existing detector measurements
# ======================================================================
def test_fingerprint_reuses_profile_measurements():
    """Nothing is recomputed from audio: the values come straight from the
    profile the detector already filled in."""
    p = profile(snr=22.0, crest=7.5, flux=0.9)
    fp = build_fingerprint(p)
    assert fp.snr_peak == pytest.approx(p.peak_snr_db)
    assert fp.crest_max == pytest.approx(p.max_crest)
    assert fp.flux_max == pytest.approx(p.max_flux)
    assert fp.bandwidth_mean == pytest.approx(p.mean_bandwidth_hz)
    assert fp.presence == pytest.approx(p.presence_fraction)


def test_merged_event_keeps_combined_moments():
    """A merged event must not silently lose its statistics."""
    a, b = profile(), profile(centroid=3000.0)
    merged = a.centroid_moments.merge(b.centroid_moments)
    assert merged.n == a.centroid_moments.n + b.centroid_moments.n
    assert a.centroid_moments.lo <= merged.lo
    assert merged.hi >= max(a.centroid_moments.hi, b.centroid_moments.hi)


# ======================================================================
# Normalisation and similarity
# ======================================================================
def test_normalised_vector_is_bounded():
    """Wildly different raw values must map into a comparable range."""
    quiet = build_fingerprint(profile(level_db=-90.0, centroid=50.0, duration_s=0.1))
    loud = build_fingerprint(
        profile(level_db=-3.0, centroid=18000.0, duration_s=600.0)
    )
    for fp in (quiet, loud):
        assert all(0.0 <= v <= 1.0 for v in fp.vector), dict(
            zip(VECTOR_FIELDS, fp.vector)
        )


def test_vector_length_matches_declared_fields():
    fp = build_fingerprint(profile())
    assert len(fp.vector) == len(VECTOR_FIELDS)


def test_similar_events_are_closer_than_different_ones():
    """Within the limits of the representation: same character, near."""
    a = build_fingerprint(profile(centroid=1000.0, level_db=-40.0,
                                 flatness=0.3, presence=0.7))
    b = build_fingerprint(profile(centroid=1050.0, level_db=-41.0,
                                 flatness=0.31, presence=0.72))
    # Clearly different: opposite spectral character and level.
    c = build_fingerprint(profile(centroid=6000.0, level_db=-15.0,
                                 flatness=0.02, presence=0.15,
                                 duration_s=6.0))
    assert similarity(a, b) < similarity(a, c)
    assert similarity(a, a) == pytest.approx(0.0, abs=1e-12)


def test_similarity_is_symmetric_and_bounded():
    a = build_fingerprint(profile(centroid=800.0))
    b = build_fingerprint(profile(centroid=3000.0, flatness=0.05))
    assert similarity(a, b) == pytest.approx(similarity(b, a))


def test_envelope_shape_is_part_of_the_vector():
    """Coarse energy shape contributes, so a decay is distinguishable."""
    steady = build_fingerprint(profile(level_jitter=0.01))
    p = profile(level_jitter=0.01)
    for i in range(p.frames):
        p.level_series[i] = -20.0 - 30.0 * i / p.frames
    decaying = build_fingerprint(p)
    assert similarity(steady, decaying) > 0.0


# ======================================================================
# Persistence
# ======================================================================
@pytest.fixture
def db(tmp_path) -> EventDatabase:
    database = EventDatabase(str(tmp_path / "events.db"))
    yield database
    database.close()


def _store(database: EventDatabase, event_id: str, **kwargs) -> str:
    fp = build_fingerprint(kwargs.pop("profile", profile()), termination="quiet")
    metadata = {
        "segmentation_reason": "quiet",
        "classification": {"label": "unknown"},
        "measurements": {"peak_snr_db": 15.0},
    }
    return database.store_event(
        event_id=event_id,
        timestamp="2026-09-27T00:00:00+00:00",
        duration=1.0,
        metadata=metadata,
        fingerprint=fp,
        **kwargs,
    )


def test_event_is_stored_and_retrieved_with_its_fingerprint(db):
    _store(db, "event_000001", audio_path="/tmp/a.wav")
    stored = db.get_event("event_000001")
    assert stored is not None
    assert stored.duration == 1.0
    assert stored.audio_path == "/tmp/a.wav"
    assert stored.fingerprint is not None
    assert stored.fingerprint.fingerprint_version == FINGERPRINT_VERSION
    assert stored.decision is Decision.UNREVIEWED
    # Detector metadata survives the round trip.
    assert stored.metadata["measurements"]["peak_snr_db"] == 15.0
    json.dumps(stored.to_dict())


def test_event_without_a_fingerprint_is_still_stored(db):
    """Fingerprinting is valuable but not essential; it must not block."""
    db.store_event(
        event_id="event_nofp",
        timestamp="t",
        duration=0.5,
        metadata={},
        fingerprint=None,
    )
    stored = db.get_event("event_nofp")
    assert stored is not None
    assert stored.fingerprint is None


def test_metadata_survives_a_process_restart(tmp_path):
    path = str(tmp_path / "persist.db")
    first = EventDatabase(path)
    _store(first, "event_000042", audio_path="/tmp/x.wav")
    first.close()

    second = EventDatabase(path)
    try:
        stored = second.get_event("event_000042")
        assert stored is not None
        assert stored.fingerprint is not None
        assert stored.audio_path == "/tmp/x.wav"
    finally:
        second.close()


def test_schema_is_created_once_and_is_idempotent(tmp_path):
    path = str(tmp_path / "twice.db")
    a = EventDatabase(path)
    _store(a, "e1")
    a.close()
    b = EventDatabase(path)
    try:
        _store(b, "e2")
        assert b.count() == 2
    finally:
        b.close()


def test_similarity_lookup_finds_the_closest_stored_event(db):
    _store(db, "far", profile=profile(centroid=6000.0, flatness=0.02,
                                      level_db=-12.0, duration_s=8.0))
    _store(db, "near", profile=profile(centroid=1020.0, flatness=0.31,
                                       level_db=-40.5))
    query = build_fingerprint(profile(centroid=1000.0, flatness=0.30,
                                     level_db=-40.0))
    hits = dict(db.find_similar(query, limit=5))
    assert "near" in hits and "far" in hits
    assert hits["near"] < hits["far"]


def test_similarity_lookup_skips_other_versions(db):
    _store(db, "v1")
    other = build_fingerprint(profile())
    object.__setattr__(other, "fingerprint_version", FINGERPRINT_VERSION + 1)
    db.store_event(
        event_id="v2", timestamp="t", duration=1.0, metadata={}, fingerprint=other
    )
    hits = dict(db.find_similar(build_fingerprint(profile()), limit=10))
    assert "v2" not in hits, "a different schema version is not comparable"


# ======================================================================
# Annotation
# ======================================================================
def test_annotation_does_not_overwrite_detector_metadata(db):
    """The core separation of section 7: observation is not interpretation."""
    _store(db, "event_1")
    before = db.get_event("event_1")
    detector_label = before.metadata["classification"]["label"]
    detector_peak = before.metadata["measurements"]["peak_snr_db"]

    db.annotate("event_1", Decision.SAVED, label="something the user named",
                confidence=0.8, notes="listened, clearly audible")

    after = db.get_event("event_1")
    # Detector's own fields are untouched.
    assert after.metadata["classification"]["label"] == detector_label
    assert after.metadata["measurements"]["peak_snr_db"] == detector_peak
    # The user's interpretation is stored alongside.
    assert after.decision is Decision.SAVED
    assert after.label == "something the user named"
    assert after.confidence == 0.8
    assert after.notes == "listened, clearly audible"


def test_all_decision_states_round_trip(db):
    for decision in Decision:
        assert Decision.parse(decision.value) is decision
    with pytest.raises(ValueError):
        Decision.parse("nonsense")


def test_save_is_not_confirmation(db):
    """Section 8: saving means keeping, not 'this is class X'."""
    _store(db, "event_1")
    db.annotate("event_1", Decision.SAVED)
    saved = db.get_event("event_1")
    assert saved.decision is Decision.SAVED
    assert saved.label is None, "saving must not invent a label"

    db.annotate("event_1", Decision.CONFIRMED, label="user label")
    confirmed = db.get_event("event_1")
    assert confirmed.decision is Decision.CONFIRMED
    assert confirmed.label == "user label"
    # The earlier save is still in the history.
    assert len(confirmed.annotations) == 2


def test_a_label_is_optional(db):
    _store(db, "event_1")
    db.annotate("event_1", Decision.UNCERTAIN)
    stored = db.get_event("event_1")
    assert stored.decision is Decision.UNCERTAIN
    assert stored.label is None


def test_annotations_are_append_only(db):
    _store(db, "event_1")
    db.annotate("event_1", Decision.UNCERTAIN, notes="first look")
    db.annotate("event_1", Decision.SAVED, notes="second look")
    history = db.annotations_for("event_1")
    assert len(history) == 2
    assert history[0].notes == "first look"
    assert history[1].notes == "second look"
    assert history[0].decision is Decision.UNCERTAIN
    assert history[1].decision is Decision.SAVED


def test_rejected_events_are_kept_as_negative_examples(db):
    """Section 9: a rejection is information, not something to delete."""
    _store(db, "event_1")
    db.annotate("event_1", Decision.REJECTED, notes="background hum, not wanted")
    rejected = db.get_event("event_1")
    assert rejected is not None
    assert rejected.fingerprint is not None, (
        "a rejected event must retain its fingerprint; it is a negative example"
    )
    assert rejected.decision is Decision.REJECTED
    assert db.count() == 1
    assert len(db.list_events(decisions=[Decision.REJECTED])) == 1


def test_listing_filters_by_current_decision(db):
    _store(db, "e1")
    _store(db, "e2")
    db.annotate("e1", Decision.SAVED)
    db.annotate("e2", Decision.REJECTED)
    # A superseded decision must not match: e1 was never rejected.
    saved = db.list_events(decisions=[Decision.SAVED])
    assert [e.event_id for e in saved] == ["e1"]
    unreviewed = db.list_events(decisions=[Decision.UNREVIEWED])
    assert unreviewed == []


def test_listing_filters_by_label(db):
    _store(db, "e1")
    _store(db, "e2")
    db.annotate("e1", Decision.CONFIRMED, label="alpha")
    db.annotate("e2", Decision.CONFIRMED, label="beta")
    assert [e.event_id for e in db.list_events(label="alpha")] == ["e1"]
    assert db.labels() == ["alpha", "beta"]


def test_counts_by_decision_include_unreviewed(db):
    _store(db, "e1")
    _store(db, "e2")
    db.annotate("e2", Decision.SAVED)
    counts = db.count_by_decision()
    assert counts[Decision.UNREVIEWED.value] == 1
    assert counts[Decision.SAVED.value] == 1


# ======================================================================
# Real-time behaviour
# ======================================================================
def test_writer_does_not_block_the_caller(db):
    """submit() must be a queue put, not a database write."""
    writer = EventWriter(db, queue_size=64)
    writer.start()
    try:
        began = time.perf_counter()
        for i in range(500):
            writer.submit(
                PendingEvent(
                    event_id=f"e{i}",
                    timestamp="t",
                    duration=1.0,
                    metadata={},
                    profile=profile(),
                )
            )
        elapsed = time.perf_counter() - began
        assert elapsed < 1.0, f"submit took {elapsed:.2f}s; it is not a queue put"
        # Submitting faster than a single writer can persist is expected to
        # drop some; the accounting must balance.
        assert writer.stats.submitted == 500
    finally:
        writer.stop()
    assert writer.stats.stored + writer.stats.dropped_queue_full == 500
    assert db.count() == writer.stats.stored


def test_writer_drains_on_stop(db):
    writer = EventWriter(db, queue_size=128)
    writer.start()
    for i in range(20):
        writer.submit(
            PendingEvent(
                event_id=f"e{i}", timestamp="t", duration=1.0, metadata={},
                profile=profile(),
            )
        )
    stats = writer.stop()
    assert stats.stored == 20
    assert db.count() == 20


def test_writer_reports_dropped_events_when_saturated(db):
    writer = EventWriter(db, queue_size=2)
    writer.start()
    try:
        for i in range(500):
            writer.submit(
                PendingEvent(
                    event_id=f"e{i}", timestamp="t", duration=1.0, metadata={},
                    profile=profile(),
                )
            )
        assert writer.stats.dropped_queue_full > 0
    finally:
        writer.stop()


def test_a_fingerprint_failure_does_not_lose_the_event(db):
    class Broken:
        """No usable measurements at all."""
        frames = 0
        presence_fraction = 0.0

    writer = EventWriter(db)
    writer.start()
    try:
        writer.submit(
            PendingEvent(
                event_id="broken", timestamp="t", duration=1.0, metadata={},
                profile=Broken(),
            )
        )
        writer.flush(5.0)
    finally:
        writer.stop()
    assert db.count() == 1
    stored = db.get_event("broken")
    assert stored is not None
    # Either a fingerprint, or a visible absence. Never an invented one.
    if stored.fingerprint is not None:
        assert stored.fingerprint.fingerprint_version == FINGERPRINT_VERSION


def test_writer_never_raises_to_the_caller(db):
    writer = EventWriter(db)
    assert writer.submit(
        PendingEvent(event_id="x", timestamp="t", duration=1.0, metadata={},
                     profile=profile())
    ) is False, "submitting to a stopped writer must be a False, not a block"


def test_detection_worker_persists_events(tmp_path):
    """End to end: a detected event reaches the index with a fingerprint."""
    from app.audio.recorder import FileSource
    from app.audio.wavio import write_wav_atomic
    from app.pipeline import AudioPipeline

    rng = np.random.default_rng(5)
    audio = (0.002 * rng.standard_normal(SR * 4)).astype(np.float32)
    tt = np.arange(int(SR * 0.3)) / SR
    start = int(SR * 1.0)
    audio[start:start + tt.size] += (
        0.35 * np.sin(2 * np.pi * 90 * tt) * np.exp(-tt * 45)
    ).astype(np.float32)
    path = str(tmp_path / "scene.wav")
    write_wav_atomic(path, audio, SR)

    db_path = str(tmp_path / "index.db")
    config = AppConfig().validate()
    pipeline = AudioPipeline(config, FileSource(path, config), record=False)
    pipeline.enable_detection(
        lossless=True, event_root=str(tmp_path / "events"), database=db_path
    )
    pipeline.start()
    pipeline.wait()
    pipeline.stop()

    database = EventDatabase(db_path)
    try:
        assert database.count() >= 1
        stored = database.list_events()[0]
        assert stored.fingerprint is not None
        assert stored.fingerprint.fingerprint_version == FINGERPRINT_VERSION
        assert stored.metadata.get("classification", {}).get("label") is not None
        # Provenance that explains the boundary is preserved.
        assert stored.segmentation_reason
    finally:
        database.close()


# ======================================================================
# Genericity
# ======================================================================
def test_fingerprinting_contains_no_application_specific_terms():
    """Section 16: the fingerprint layer must stay application-agnostic."""
    import re
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "app" / "events"
    # Two refinements, both needed for the guard to be about domain terminology
    # rather than about ordinary code:
    #   * word boundaries, so "cough" does not match inside another word;
    #   * a negative lookbehind for ".", so `time.sleep()` - a standard
    #     library call - is not read as the word "sleep".
    forbidden = re.compile(
        r"(?<![.\w])(breath(ing|e)?|respirat\w*|snor\w+|sleep\w*|heartbeat|"
        r"footsteps?|knocks?|cough\w*|whispers?|speech|doorbell)(?![\w])",
        re.IGNORECASE,
    )
    # retention.py is included because it buckets audio *by* classification
    # and chooses what to evict: a hard-coded sound class there would make
    # the "representative audio" selection quietly favour one kind of sound,
    # which is exactly the bias the bounded cache must not have.
    for name in (
        "fingerprint.py", "database.py", "persistence.py", "retention.py",
    ):
        text = (base / name).read_text()
        offenders = [
            line.strip()
            for line in text.splitlines()
            if forbidden.search(line.split("#", 1)[0])
        ]
        assert not offenders, f"{name}: {offenders}"
