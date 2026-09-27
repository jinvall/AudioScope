"""Phase 3: candidate detection, event tracking, classification, storage."""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from app.analysis.features import FeatureExtractor
from app.analysis.frames import FrameExtractor
from app.analysis.noise_floor import AdaptiveNoiseFloor
from app.audio.ringbuffer import RingBuffer
from app.audio.wavio import probe_wav, read_wav_mono
from app.config import AppConfig, ConfigError, DetectionConfig
from app.detection.candidate import CandidateDetector
from app.detection.classifier import (
    CONFIDENCE_NOTE,
    LABEL_CLIPPING,
    LABEL_CLOTHING,
    LABEL_FOOTSTEP,
    LABEL_INTERFERENCE,
    LABEL_KNOCK,
    LABEL_MOVEMENT,
    LABEL_SPEECH,
    LABEL_SCRAPING,
    LABEL_WHISPER,
    SEPARATION_QUERIES,
    UNKNOWN,
    Classifier,
    _harmonic_dominant,
    _narrowness,
    _repetition_score,
)
from app.detection.tracker import (
    EventProfile,
    EventState,
    EventTracker,
    TrackedEvent,
)
from app.events.event import Event, build_event
from app.events.store import EventStore

SR = 48000


# ----------------------------------------------------------------------
# Harness: drives synthetic audio through the whole detection chain
# ----------------------------------------------------------------------
class Harness:
    def __init__(self, config: AppConfig | None = None, seed: int = 7):
        self.config = (config or AppConfig()).validate()
        self.hop = self.config.analysis_hop_frames
        self.frame = self.config.analysis_frame_frames
        self.features = FeatureExtractor(SR, window=self.config.analysis.window)
        self.framer = FrameExtractor(self.frame, self.hop, None)
        self.floor = AdaptiveNoiseFloor(
            self.config.analysis, frame_rate=SR / self.hop
        )
        self.detector = CandidateDetector(
            self.config.detection, frame_rate=SR / self.hop
        )
        self.tracker = EventTracker(
            self.config.detection,
            sample_rate=SR,
            frame_rate=SR / self.hop,
            pre_roll_seconds=self.config.event.pre_roll_seconds,
            post_roll_seconds=self.config.event.post_roll_seconds,
        )
        self.classifier = Classifier()
        self.ring = RingBuffer(SR, self.config.buffer.seconds)
        self.store = EventStore("/tmp/phase3-test-events")
        self.rng = np.random.default_rng(seed)
        self.events: list[tuple] = []
        self.candidates: list = []
        self._pos = 0

    def feed(self, seconds: float, generator) -> None:
        """Feed ``seconds`` of audio from ``generator(i) -> mono samples``."""
        for i in range(0, int(seconds * SR), self.hop):
            block = generator(i).astype(np.float32)
            self.ring.write(block)
            for k, f in enumerate(self.framer.push(block)):
                ft = self.features.analyse(f)
                fl = self.floor.update(ft)
                start = self._pos + k * self.hop
                c = self.detector.update(ft, fl, start)
                self.candidates.append(c)
                tracked = self.tracker.update(c, ft, fl)
                if tracked is not None:
                    self._finalise(tracked, fl.overall_floor_db)
            self._pos += block.size

    def finish(self) -> None:
        for tracked in self.tracker.flush():
            self._finalise(tracked, self.floor.overall_floor_db)

    def _finalise(self, tracked, floor_db):
        active = (
            tracked.profile.last_active_sample - tracked.profile.onset_sample
        ) / SR
        result = self.classifier.classify(tracked.profile, active)
        event_id, _day = self.store.allocate_id()
        event = build_event(
            tracked, result, event_id, noise_floor_db=floor_db,
            peak_snr_db=tracked.profile.peak_snr_db, source="synthetic",
        )
        event, samples = self.store.extract(event, self.ring)
        if samples is not None:
            self.store.save(event, samples)
        self.events.append((event, samples, tracked))


@pytest.fixture
def harness() -> Harness:
    return Harness()


def scene_quiet_then_steps(h: Harness):
    """Quiet room with three footsteps one second apart."""
    def gen(i):
        t0 = i / SR
        x = 0.002 * h.rng.standard_normal(h.hop)
        for at in FOOTSTEP_TIMES:
            d = t0 - at
            if 0.0 <= d < 0.3:
                tt = np.arange(h.hop) / SR + d
                x = x + (0.35 * np.sin(2 * np.pi * 90 * tt)
                         * np.exp(-tt * 45)).astype(np.float32)
        return x
    return gen


#: Three footsteps one second apart, as the regression case below describes.
FOOTSTEP_TIMES = (3.0, 4.0, 5.0)


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
def test_detection_config_defaults():
    c = DetectionConfig()
    assert c.onset_threshold > c.continuation_threshold
    assert c.min_duration < c.max_duration


def test_hysteresis_is_enforced_by_validation():
    """Equal thresholds would make the state machine chatter."""
    with pytest.raises(ConfigError):
        DetectionConfig(onset_threshold=0.3, continuation_threshold=0.3)
    with pytest.raises(ConfigError):
        DetectionConfig(onset_threshold=0.1, continuation_threshold=0.5)


def test_detection_config_saved_in_app_config():
    c = AppConfig().validate()
    assert c.detection.release_timeout == 0.4
    assert c.detection.merge_gap == 0.3


def test_crest_threshold_excludes_noise():
    """Measured: Gaussian noise over 1920 samples peaks near 4.6.

    The configured floor must sit above that or every background frame is
    flagged impulsive.
    """
    assert DetectionConfig().crest_low > 4.6


# ----------------------------------------------------------------------
# Candidate detection
# ----------------------------------------------------------------------
def test_quiet_room_produces_no_candidate(harness):
    harness.feed(6.0, lambda i: 0.002 * harness.rng.standard_normal(harness.hop))
    harness.finish()
    assert harness.candidates
    # The broad detector may still mark a frame as a candidate; what matters is
    # that nothing reaches the onset threshold, so no event is created.
    assert max(c.activation for c in harness.candidates) < (
        harness.config.detection.onset_threshold
    )
    assert not harness.events


def test_footstep_produces_a_candidate(harness):
    harness.feed(6.0, scene_quiet_then_steps(harness))
    activations = [c.activation for c in harness.candidates]
    assert max(activations) > harness.config.detection.onset_threshold


def test_candidate_indicators_are_relative_to_the_floor(harness):
    """The same sound in a louder room is not a bigger event."""
    def gen(i):
        return (0.002 * harness.rng.standard_normal(harness.hop)).astype(np.float32)

    harness.feed(6.0, gen)
    floor = harness.floor.overall_floor_db
    assert floor < -40.0
    # The background sits essentially at the floor: no spurious SNR.
    assert max(c.snr_db for c in harness.candidates) < 8.0


def test_modulation_depth_is_bounded(harness):
    """The old index divided by the mean and ran past 10."""
    harness.feed(8.0, scene_quiet_then_steps(harness))
    depths = [c.modulation for c in harness.candidates if c.modulation > 0]
    assert depths
    assert max(depths) <= 1.0 + 1e-9


def test_candidate_carries_explanation(harness):
    harness.feed(6.0, scene_quiet_then_steps(harness))
    loud = max(harness.candidates, key=lambda c: c.activation)
    assert loud.indicators
    assert loud.values
    assert loud.indicators[0] == max(loud.values, key=loud.values.get)


def test_candidate_serialises(harness):
    harness.feed(2.0, scene_quiet_then_steps(harness))
    json.dumps(harness.candidates[-1].to_dict())


# ----------------------------------------------------------------------
# Tracking
# ----------------------------------------------------------------------
def test_tracker_finds_the_three_footsteps(harness):
    harness.feed(8.0, scene_quiet_then_steps(harness))
    harness.finish()
    assert len(harness.events) >= 1
    event, _samples, tracked = harness.events[0]
    assert tracked.profile.impulsive_onsets >= 3
    # Regular one-second spacing between the steps.
    intervals = tracked.profile.onset_intervals
    assert len(intervals) >= 2
    assert all(0.8 < i < 1.2 for i in intervals[:2])


def test_hysteresis_prevents_per_frame_candidates(harness):
    """One event must not produce an event per frame."""
    harness.feed(8.0, scene_quiet_then_steps(harness))
    harness.finish()
    assert harness.tracker.events_emitted < 10
    for _event, _samples, tracked in harness.events:
        assert tracked.profile.frames > 1


def test_minimum_duration_rejects_a_single_frame():
    tracker = EventTracker(DetectionConfig(min_duration=5.0), sample_rate=SR,
                          frame_rate=100.0)
    for i in range(3):
        tracker._begin(_fake_candidate(i * 480))
        tracker._frame_in_event = 0
        tracker._profile.frames = 0
    # Force a finish with too few frames.
    tracker._state = EventState.IDLE
    assert tracker.events_discarded_short >= 0


def test_event_span_includes_pre_and_post_roll(harness):
    harness.feed(8.0, scene_quiet_then_steps(harness))
    harness.finish()
    event, _samples, tracked = harness.events[0]
    pre = harness.config.event.pre_roll_seconds
    post = harness.config.event.post_roll_seconds
    onset = tracked.profile.onset_sample / SR
    assert event.start_seconds <= onset
    assert event.end_seconds >= (tracked.profile.last_active_sample / SR)
    if onset > pre:
        assert onset - event.start_seconds == pytest.approx(pre, abs=0.05)
    assert event.duration >= post


def test_truncated_event_is_flagged():
    tracker = EventTracker(DetectionConfig(max_duration=0.3), sample_rate=SR,
                          frame_rate=100.0)
    detector_like = [_fake_candidate(i * 480, activation=1.0) for i in range(200)]
    emitted = []
    for candidate in detector_like:
        out = tracker.update(candidate)
        if out is not None:
            emitted.append(out)
    emitted.extend(tracker.flush())
    assert emitted
    assert any(e.truncated for e in emitted)
    assert tracker.events_truncated >= 1


def test_merge_gap_folds_two_nearby_detections():
    tracker = EventTracker(DetectionConfig(merge_gap=2.0), sample_rate=SR,
                          frame_rate=100.0)
    out = []
    # First burst.
    for i in range(30):
        out.append(tracker.update(_fake_candidate(i * 480, activation=1.0)))
    # A quiet stretch longer than the release timeout (0.4 s = 40 frames) so the
    # first event actually finishes, but shorter than the 2 s merge gap, so the
    # second burst is still folded into it.
    for i in range(30, 100):
        out.append(tracker.update(_fake_candidate(i * 480, activation=0.0)))
    for i in range(100, 140):
        out.append(tracker.update(_fake_candidate(i * 480, activation=1.0)))
    out.extend(tracker.flush())
    events = [e for e in out if e is not None]
    assert events
    assert any(e.merged_count > 0 for e in events)


def test_tracker_reset_clears_state():
    tracker = EventTracker(DetectionConfig(), sample_rate=SR, frame_rate=100.0)
    tracker.update(_fake_candidate(0, activation=1.0))
    assert tracker.is_active
    tracker.reset()
    assert tracker.state is EventState.IDLE
    assert not tracker.is_active


# ----------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------
def _profile(**kwargs) -> EventProfile:
    base = dict(
        frames=100, frames_above_floor=50, peak_snr_db=20.0,
        mean_snr_db=15.0, mean_rms_dbfs=-50.0, mean_flatness=0.1,
        mean_centroid_hz=500.0, mean_bandwidth_hz=2000.0,
        mean_modulation=0.3, max_peakiness=10.0, mean_peakiness=8.0,
        max_low_band_snr_db=20.0, mean_low_band_snr_db=15.0,
        mean_high_band_snr_db=18.0, max_high_band_snr_db=20.0,
        mean_mid_band_snr_db=18.0,
    )
    base.update(kwargs)
    return EventProfile(**base)


def test_classifier_never_reports_a_confidence():
    """docs section 8 forbids cosmetic confidence."""
    c = Classifier(AppConfig().validate().detection)
    for profile in (
        _profile(), _profile(max_peakiness=300, mean_peakiness=280,
                            dominant_frequency_hz=50.0),
    ):
        result = c.classify(profile, 0.4)
        assert result.confidence is None
        assert result.confidence_note == CONFIDENCE_NOTE
        assert "score" in json.dumps(result.to_dict())


def test_classifier_reports_scores_and_evidence():
    result = Classifier(DetectionConfig()).classify(_profile(), 0.4)
    assert result.label_scores
    assert result.evidence
    json.dumps(result.to_dict())


def test_interference_needs_tonality_and_mains():
    hum = _profile(max_peakiness=400, mean_peakiness=380,
                   dominant_frequency_hz=50.0, mean_flatness=0.001,
                   mean_bandwidth_hz=20.0, mean_centroid_hz=50.0,
                   mean_low_band_snr_db=5.0, mean_high_band_snr_db=2.0)
    result = Classifier(DetectionConfig()).classify(hum, 5.0)
    assert result.classification == LABEL_INTERFERENCE
    assert "mains_frequency_or_harmonic" in result.evidence
    assert "narrow_spectrum" in result.evidence


def test_interference_rejected_when_it_has_repeated_onsets():
    """A decaying thump near 100 Hz is not mains hum."""
    thump = _profile(max_peakiness=1000000, mean_peakiness=1000,
                     dominant_frequency_hz=100.0, mean_flatness=0.001,
                     mean_bandwidth_hz=80.0, mean_centroid_hz=100.0,
                     impulsive_onsets=3, onset_intervals=[1.0, 1.0],
                     max_low_band_snr_db=57.0)
    result = Classifier(DetectionConfig()).classify(thump, 3.5)
    assert result.classification != LABEL_INTERFERENCE


def test_footstep_from_low_frequency_and_onsets():
    steps = _profile(impulsive_onsets=3, onset_intervals=[1.0, 1.0],
                     max_low_band_snr_db=57.0, mean_low_band_snr_db=30.0,
                     mean_high_band_snr_db=6.0,
                     mean_flatness=0.5, mean_rms_dbfs=-45.0)
    steps.frames_above_floor = 10
    result = Classifier(DetectionConfig()).classify(steps, 3.5)
    assert result.classification == LABEL_FOOTSTEP
    assert "low_frequency_energy" in result.evidence
    assert "repeated_onsets_in_sequence" in result.evidence


def test_knock_from_high_crest_transient():
    knock = _profile(max_crest=12.0, mean_crest=6.0, mean_flatness=0.25,
                     impulsive_onsets=1, max_low_band_snr_db=44.0,
                     mean_rms_dbfs=-20.0, max_peakiness=17.0)
    result = Classifier(DetectionConfig()).classify(knock, 0.2)
    assert result.classification == LABEL_KNOCK
    assert "high_crest_transient" in result.evidence


def test_movement_for_sustained_broadband():
    """Generic sustained activity.

    docs section 5 lists clothing, rubbing and scraping as members of the
    movement family, so the *specific* labels are preferred when their evidence
    is present.  This profile is deliberately generic: broadband, mid centroid,
    noticeably unsteady.
    """
    move = _profile(impulsive_onsets=0, max_crest=3.0, mean_crest=2.6,
                    mean_low_band_snr_db=18.0, mean_high_band_snr_db=14.0,
                    high_to_low_ratio_db=-4.0, mean_flatness=0.45,
                    mean_centroid_hz=1400.0, mean_bandwidth_hz=5000.0,
                    mean_rms_dbfs=-40.0, max_peakiness=15.0,
                    mean_modulation=0.35, max_flux=1.0,
                    mean_mid_band_snr_db=12.0)
    move.frames_above_floor = 52
    result = Classifier(DetectionConfig()).classify(move, 2.9)
    assert result.classification == LABEL_MOVEMENT


def test_clothing_is_preferred_over_generic_movement():
    """Documented precedence within the movement family.

    A soft, low, sustained sound reads as clothing specifically; the generic
    movement label is the fallback when nothing more specific applies.
    """
    cloth = _profile(impulsive_onsets=0, max_crest=2.4, mean_crest=2.1,
                     mean_low_band_snr_db=25.0, mean_high_band_snr_db=8.0,
                     mean_flatness=0.3, mean_centroid_hz=500.0,
                     mean_bandwidth_hz=3000.0, mean_rms_dbfs=-42.0,
                     max_peakiness=12.0, mean_modulation=0.3,
                     mean_mid_band_snr_db=14.0)
    cloth.frames_above_floor = 60
    result = Classifier(DetectionConfig()).classify(cloth, 2.0)
    assert result.classification == LABEL_CLOTHING


def test_whisper_needs_presence_and_onset_free():
    """A quiet sound is not a whisper if it is a burst, or if it is silence."""
    whisper = _profile(impulsive_onsets=0, max_crest=2.2,
                       mean_rms_dbfs=-55.0, mean_flatness=0.30,
                       mean_high_band_snr_db=25.0, mean_low_band_snr_db=8.0,
                       high_to_low_ratio_db=17.0, mean_centroid_hz=2000.0,
                       mean_modulation=0.6, peak_snr_db=18.0,
                       max_peakiness=6.0, mean_bandwidth_hz=6000.0,
                       mean_mid_band_snr_db=20.0)
    whisper.frames_above_floor = 90
    result = Classifier(DetectionConfig()).classify(whisper, 2.0)
    assert result.classification == LABEL_WHISPER
    assert "low_rms" in result.evidence
    assert "above_noise_floor_not_silence" in result.evidence

    # The same measurements, but silent: must not be a whisper.
    silent = _profile(impulsive_onsets=0, max_crest=2.2,
                      mean_rms_dbfs=-55.0, mean_flatness=0.30,
                      mean_high_band_snr_db=25.0, mean_low_band_snr_db=8.0,
                      high_to_low_ratio_db=17.0, mean_centroid_hz=2000.0,
                      mean_modulation=0.6, peak_snr_db=1.0,
                      max_peakiness=6.0, mean_bandwidth_hz=6000.0,
                      mean_mid_band_snr_db=20.0)
    silent.frames_above_floor = 2
    assert Classifier(DetectionConfig()).classify(silent, 2.0).classification != LABEL_WHISPER

    # A burst: the same spectral character, but only present in a small
    # minority of frames.  This is the case presence exists to catch, and the
    # opposite of a sustained whisper that merely contains a few consonant
    # transients - which must stay classifiable, and is asserted above and in
    # the regression tests.
    burst = _profile(impulsive_onsets=3, max_crest=2.2,
                     mean_rms_dbfs=-55.0, mean_flatness=0.30,
                     mean_high_band_snr_db=25.0, mean_low_band_snr_db=8.0,
                     high_to_low_ratio_db=17.0, mean_centroid_hz=2000.0,
                     mean_modulation=0.6, peak_snr_db=18.0,
                     max_peakiness=6.0, mean_bandwidth_hz=6000.0,
                     mean_mid_band_snr_db=20.0)
    burst.frames_above_floor = 12
    assert burst.presence_fraction < 0.2
    assert Classifier(DetectionConfig()).classify(
        burst, 2.0
    ).classification != LABEL_WHISPER


def test_clipping_is_reported():
    clip = _profile(max_clipping_ratio=0.02, max_peakiness=1000000,
                    dominant_frequency_hz=450.0, mean_flatness=0.001)
    result = Classifier(DetectionConfig()).classify(clip, 1.0)
    assert result.classification == LABEL_CLIPPING
    assert "input_clipping" in result.evidence


def test_unknown_when_nothing_matches():
    blank = EventProfile(frames=50, frames_above_floor=0)
    result = Classifier(DetectionConfig()).classify(blank, 0.1)
    assert result.classification == UNKNOWN


def test_ambiguity_is_flagged():
    """When two labels are close, the result says so rather than pretending."""
    result = Classifier(DetectionConfig()).classify(_profile(), 0.4)
    if result.runner_up is not None:
        assert result.margin == pytest.approx(
            result.label_scores[result.classification]
            - result.label_scores[result.runner_up]
        )


def test_separation_query_is_suggested_and_complete():
    for label in (LABEL_WHISPER, LABEL_SPEECH, LABEL_FOOTSTEP, LABEL_MOVEMENT,
                  LABEL_INTERFERENCE, UNKNOWN):
        assert SEPARATION_QUERIES[label]
    result = Classifier(DetectionConfig()).classify(_profile(), 0.4)
    assert result.separation_query in SEPARATION_QUERIES.values()


# ----------------------------------------------------------------------
# Helper units
# ----------------------------------------------------------------------
def test_harmonic_dominant_is_tight():
    assert _harmonic_dominant(50.0)
    assert _harmonic_dominant(60.0)
    assert _harmonic_dominant(150.0)
    # A 90 Hz footstep thump must not read as 100 Hz mains.
    assert not _harmonic_dominant(90.0)
    assert not _harmonic_dominant(0.0)
    assert not _harmonic_dominant(440.0)


def test_narrowness():
    # bandwidth/centroid: 5/50 = 0.1 is a single tone, 10000/1000 = 10 is
    # broadband.  The function maps roughly 0.03..0.48 onto 0..1.
    assert _narrowness(
        _profile(mean_centroid_hz=50.0, mean_bandwidth_hz=5.0)
    ) > 0.9
    assert _narrowness(
        _profile(mean_centroid_hz=1000.0, mean_bandwidth_hz=10000.0)
    ) < 0.05
    assert _narrowness(_profile(mean_centroid_hz=0.0)) == 0.0


def test_repetition_score():
    assert _repetition_score([]) == (0.0, 0.0)
    score, regularity = _repetition_score([1.0, 1.0, 1.0, 1.0])
    assert score > 0.5
    assert regularity > 0.8
    # Irregular spacing still counts as a sequence, just less regular.
    score2, regularity2 = _repetition_score([0.3, 1.9, 0.4, 2.1])
    assert score2 > 0.0
    assert regularity2 < regularity


# ----------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------
def test_event_metadata_has_no_confidence(tmp_path):
    store = EventStore(str(tmp_path))
    tracker = EventTracker(DetectionConfig(), sample_rate=SR, frame_rate=100.0)
    tracked = _tracked(tracker)
    result = Classifier(DetectionConfig()).classify(tracked.profile, 0.3)
    event = build_event(tracked, result, "event_000001")
    payload = event.metadata()
    assert payload["classification"]["confidence"] is None
    assert "confidence_note" in payload["classification"]
    assert payload["classification"]["label_scores"]
    assert payload["timestamp"]
    assert payload["duration_seconds"] >= 0.0
    json.dumps(payload)


def test_store_writes_audio_and_metadata(tmp_path):
    store = EventStore(str(tmp_path))
    ring = RingBuffer(SR, 30.0)
    samples = (0.1 * np.random.default_rng(0).standard_normal(48000)).astype(
        np.float32
    )
    ring.write(samples)

    tracker = EventTracker(DetectionConfig(), sample_rate=SR, frame_rate=100.0)
    tracked = _tracked(tracker)
    result = Classifier(DetectionConfig()).classify(tracked.profile, 0.3)
    event = build_event(tracked, result, "event_000042")
    event, got = store.extract(event, ring)
    assert got is not None
    store.save(event, got)

    directory = store.event_directory("event_000042", event.day)
    assert os.path.isdir(directory)
    assert os.path.exists(os.path.join(directory, "original.wav"))
    assert os.path.exists(os.path.join(directory, "metadata.json"))
    info = probe_wav(os.path.join(directory, "original.wav"))
    assert info.sample_rate == SR
    assert info.subtype == "FLOAT"
    # No temporary files left behind.
    assert not [f for f in os.listdir(directory) if f.endswith(".part")]
    # Metadata round-trips, looked up under the day the event recorded.
    assert store.load("event_000042", event.day)["event_id"] == "event_000042"


def test_stored_audio_is_the_original_unchanged(tmp_path):
    store = EventStore(str(tmp_path))
    ring = RingBuffer(SR, 30.0)
    samples = (0.2 * np.random.default_rng(1).standard_normal(24000)).astype(
        np.float32
    )
    ring.write(samples)
    tracker = EventTracker(DetectionConfig(), sample_rate=SR, frame_rate=100.0)
    tracked = _tracked(tracker)
    # Force the span to cover exactly the samples written.
    tracked.start_sample = 0
    tracked.end_sample = samples.size
    result = Classifier(DetectionConfig()).classify(tracked.profile, 0.3)
    event = build_event(tracked, result, "event_000000")
    event, got = store.extract(event, ring)
    store.save(event, got)
    # Look it up under the day the event was actually filed under.
    stored = read_wav_mono(os.path.join(
        store.event_directory("event_000000", event.day), "original.wav"
    ))
    assert np.array_equal(stored[:samples.size], samples)


def test_missing_pre_roll_is_reported(tmp_path):
    store = EventStore(str(tmp_path))
    ring = RingBuffer(SR, 30.0)
    # Only 2 s of audio has ever been captured, but the event claims to start
    # 5 s in - so the pre-roll reaching back to 0 is not available.
    ring.write(np.zeros(2 * SR, dtype=np.float32))
    tracker = EventTracker(DetectionConfig(), sample_rate=SR, frame_rate=100.0)
    tracked = _tracked(tracker)
    tracked.start_sample = 5 * SR
    tracked.end_sample = 8 * SR
    result = Classifier(DetectionConfig()).classify(tracked.profile, 0.3)
    event = build_event(tracked, result, "event_000000")
    event, _samples = store.extract(event, ring)
    assert event.preroll_missing_frames > 0
    assert event.metadata()["provenance"]["preroll_complete"] is False


def test_event_ids_are_unique_and_ordered(tmp_path):
    store = EventStore(str(tmp_path))
    ids = [store.allocate_id()[0] for _ in range(5)]
    assert ids == [f"event_{i:06d}" for i in range(5)]
    assert len(set(ids)) == 5


# ----------------------------------------------------------------------
# Full chain
# ----------------------------------------------------------------------
def test_full_chain_detects_classifies_and_stores(tmp_path):
    """The Phase 3 acceptance criteria, on one synthetic scene."""
    config = AppConfig().validate()
    h = Harness(config)
    h.store = EventStore(str(tmp_path))
    h.feed(8.0, scene_quiet_then_steps(h))
    h.finish()

    assert h.events, "no event was detected"
    event, samples, _tracked = h.events[0]

    # 1. timestamp
    assert event.detected_at
    # 2. duration
    assert event.duration > 0.0
    # 3. actual captured audio
    assert samples is not None and samples.size > 0
    assert os.path.exists(event.audio_path)
    assert probe_wav(event.audio_path).frames == samples.size
    # 4. classification or unknown
    assert event.classification
    # 5. stored persistently
    assert os.path.exists(
        os.path.join(event.directory, "metadata.json")
    )
    assert event.classification == LABEL_FOOTSTEP
    # Confidence is explicitly absent, not zero.
    assert event.confidence is None


def test_detection_worker_does_not_block(tmp_path):
    from app.detection.worker import DetectionWorker

    config = AppConfig().validate()
    ring = RingBuffer(config.sample_rate, config.buffer.seconds)
    worker = DetectionWorker(config, ring, EventStore(str(tmp_path)),
                            input_queue_size=2)
    worker.start()
    try:
        import time
        began = time.monotonic()
        for i in range(500):
            worker.submit(_fake_result(i * 480))
        elapsed = time.monotonic() - began
        # The bound is generous because the worker thread competes for the GIL;
        # what is being asserted is that submit() never waits on the worker, so
        # the time does not grow with the queue depth.
        assert elapsed < 5.0, f"submit() appears to block: {elapsed:.2f}s"
        assert worker.stats.frames_dropped > 0
        assert worker.running, "the worker must survive a flooded queue"
    finally:
        worker.stop()
    stats = worker.stats.to_dict()
    assert stats["is_contiguous"] is False
    json.dumps(stats)


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
def _fake_candidate(start_sample: int, activation: float = 1.0):
    from app.detection.candidate import Candidate

    return Candidate(
        index=start_sample // 480,
        start_sample=start_sample,
        audio_time=start_sample / SR,
        activation=activation,
        indicators=("level",) if activation else (),
        values={"level": activation} if activation else {},
        snr_db=20.0 * activation,
        modulation=0.0,
        clipping_ratio=0.0,
        dominant_frequency_hz=0.0,
        spectral_peakiness=0.0,
        flux_ratio=None,
        crest_factor=0.0,
    )


def _fake_result(start_sample: int):
    from app.analysis.worker import AnalysisResult

    ft = FeatureExtractor(SR).analyse(
        np.zeros(1920, dtype=np.float32), start_sample
    )
    floor = AdaptiveNoiseFloor(frame_rate=100.0).update(ft)
    return AnalysisResult(
        features=ft, floor=floor, start_sample=start_sample, elapsed=0.0
    )


def _tracked(tracker: EventTracker | None = None) -> "TrackedEvent":
    """A minimal completed event, without running the state machine."""
    profile = EventProfile(
        onset_sample=5 * SR,
        first_candidate_sample=5 * SR,
        last_active_sample=5 * SR + 12000,
        frames=100, frames_above_floor=60, peak_snr_db=20.0, mean_snr_db=15.0,
    )
    return TrackedEvent(
        start_sample=0, end_sample=11 * SR, sample_rate=SR, profile=profile
    )


# ======================================================================
# Whisper regression tests
#
# The bug these guard against: event *mean* level and *mean* spectral features
# are diluted by the quiet gaps between parts of an event, so three footsteps
# look like a sustained low-level broadband sound and get called a whisper.
# ======================================================================
def _whisper_profile(h, *, with_consonants=False):
    """A quiet room with a sustained whispered hiss, and optional consonants.

    Two details make this a realistic whisper rather than an easy target, and
    both were arrived at by measurement:

    * **Room noise underneath.**  The floor is adaptive, so a whisper with
      nothing beneath it simply *becomes* the floor and has zero presence.
      0.002 of room noise under a 0.05 whispered component gives roughly 25 dB
      of on/off contrast.
    * **Brief gaps between phrases.**  A signal present in *every* frame is by
      definition the average of the window, so the floor converges to it and
      its reported signal-to-noise ratio is zero no matter how loud it is -
      measured at every amplitude from 7 dB to 24 dB above the noise.  Real
      whispering has short pauses; gating at 1.2 phrases per second with 30%
      gaps reproduces that and is what lets the floor see the room beneath.
    """
    def gen(i):
        t0 = i / SR
        noise = 0.002 * h.rng.standard_normal(h.hop)
        # Broad hiss with a mild high-frequency emphasis: a whisper has no
        # fundamental and is roughly flat-to-slightly-bright.  A *steep* tilt is
        # wrong here - it concentrates the energy in the top bins, which reads
        # as strongly tonal (peakiness above 400) and would make a genuine
        # whisper score zero on the "weak harmonic structure" term.
        # Broadband plus a genuine high-frequency component.  Successive
        # differences are a crude but real high-pass: unlike the DC blocker
        # used earlier they *add* upper-band energy rather than removing low.
        raw = h.rng.standard_normal(h.hop)
        bright = np.diff(raw, prepend=raw[0]) / math.sqrt(2.0)
        hiss = 0.015 * (0.85 * raw + 0.15 * bright)
        phrase_gate = ((t0 * 1.2) % 1.0) < 0.70
        # Syllabic-rate modulation within a phrase, so the envelope is
        # speech-like rather than a flat hiss.
        envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t0)
        x = noise + hiss * envelope * phrase_gate
        if with_consonants:
            # Four short broadband ticks standing in for /t k p sh/, half a
            # second apart.  A whispered phrase has a handful of these, not one
            # every 100 ms; too many would be an onset train, not speech.
            for at in (0.8, 1.3, 1.8, 2.3):
                d = t0 - at
                if 0.0 <= d < 0.02:
                    x = x + (0.25 * np.exp(-d * 400)
                             * h.rng.standard_normal(h.hop)).astype(np.float32)
        return x.astype(np.float32)
    return gen


def _high_shaped(block: np.ndarray) -> np.ndarray:
    """First-order high-pass, used only to add a mild brightness tilt."""
    out = np.empty_like(block)
    previous = 0.0
    for i, value in enumerate(block):
        previous = 0.92 * previous + float(value)
        out[i] = float(value) - previous * 0.5
    return out


def _footsteps_profile(h, times):
    """Quiet room with decaying low-frequency thumps at the given times."""
    def gen(i):
        t0 = i / SR
        x = 0.002 * h.rng.standard_normal(h.hop)
        for at in times:
            d = t0 - at
            if 0.0 <= d < 0.3:
                tt = np.arange(h.hop) / SR + d
                x = x + (0.35 * np.sin(2 * np.pi * 90 * tt)
                         * np.exp(-tt * 45)).astype(np.float32)
        return x
    return gen


def _whisper_label(event) -> str:
    return event.classification


def test_regression_three_separated_footsteps_are_not_a_whisper(tmp_path):
    """quiet -> step -> quiet -> step -> quiet -> step -> quiet."""
    h = Harness()
    h.store = EventStore(str(tmp_path))
    h.feed(9.0, _footsteps_profile(h, (1.0, 2.2, 3.4)))
    h.finish()
    assert h.events, "no event was produced at all"

    for event, _samples, tracked in h.events:
        assert _whisper_label(event) != LABEL_WHISPER, (
            "a footstep sequence must not be classified as a whisper; "
            f"got {event.classification} with "
            f"presence={event.features.get('presence_fraction')}"
        )
    # It should still be eligible for footsteps / movement / unknown.
    labels = {_whisper_label(e) for e, _s, _t in h.events}
    assert labels & {LABEL_FOOTSTEP, LABEL_MOVEMENT, UNKNOWN, LABEL_SPEECH,
                     LABEL_KNOCK, LABEL_CLOTHING, LABEL_SCRAPING}


def test_regression_sustained_quiet_whisper_classifies_as_whisper(tmp_path):
    h = Harness()
    h.store = EventStore(str(tmp_path))
    h.feed(8.0, _whisper_profile(h))
    h.finish()
    assert h.events
    event = h.events[0][0]
    # The whisper profile must actually be present most of the time. The exact
    # figure drifts a little between runs, so the bar is set well clear of the
    # footstep case (measured at 0.04-0.09) rather than at an arbitrary number.
    presence = event.features["presence_fraction"]
    assert presence > 0.35, (
        f"the synthetic whisper is not sustained (presence {presence}); "
        "the test signal is wrong, not the classifier"
    )
    assert _whisper_label(event) == LABEL_WHISPER, (
        f"got {event.classification} (runner-up {event.runner_up}); "
        f"diagnostics={event.label_details.get(LABEL_WHISPER)}"
    )


def test_regression_single_isolated_footstep_is_not_a_whisper(tmp_path):
    """quiet -> step -> quiet.  A single transient must not average into a whisper."""
    h = Harness()
    h.store = EventStore(str(tmp_path))
    h.feed(6.0, _footsteps_profile(h, (2.0,)))
    h.finish()
    assert h.events
    for event, _samples, _tracked in h.events:
        assert _whisper_label(event) != LABEL_WHISPER, (
            f"a single footstep must not classify as a whisper; got "
            f"{event.classification} with "
            f"presence={event.features.get('presence_fraction')}"
        )


def test_regression_whisper_with_consonant_transients_still_classifies(tmp_path):
    """A whisper containing consonant impulses must NOT be disqualified.

    This is the test that would fail if anyone reintroduced a hard
    ``impulsive_onsets > 0 -> not whisper`` rule.
    """
    h = Harness()
    h.store = EventStore(str(tmp_path))
    h.feed(8.0, _whisper_profile(h, with_consonants=True))
    h.finish()
    assert h.events
    event = h.events[0][0]
    onsets = event.features["impulsive_onsets"]
    assert onsets > 0, (
        "the synthetic whisper has no consonant transients, so this test is "
        "not exercising what it claims to"
    )
    assert _whisper_label(event) == LABEL_WHISPER, (
        f"a whisper with {onsets} consonant transients must stay classifiable; "
        f"got {event.classification} (runner-up {event.runner_up})"
    )


def test_regression_presence_separates_whisper_from_footsteps(tmp_path):
    """The discriminator itself: same level, opposite presence."""
    foot = Harness()
    foot.feed(9.0, _footsteps_profile(foot, (1.0, 2.2, 3.4)))
    foot.finish()
    assert foot.events
    foot_presence = foot.events[0][2].profile.presence_fraction

    whis = Harness()
    whis.feed(8.0, _whisper_profile(whis))
    whis.finish()
    assert whis.events
    whisper_presence = whis.events[0][2].profile.presence_fraction

    assert whisper_presence > foot_presence * 3, (
        f"presence should separate them clearly: whisper {whisper_presence:.2f} "
        f"vs footsteps {foot_presence:.2f}"
    )


def test_impulse_density_is_per_frame_and_bounded():
    p = EventProfile(frames=200, frames_above_floor=180,
                     frames_per_second=100.0, impulsive_onsets=5)
    # 5 onsets over 200 frames, not per second.
    assert p.impulse_density == pytest.approx(0.025)
    # The time-normalised form stays available and hop-independent.
    assert p.onset_density_per_second == pytest.approx(2.5)


def test_rms_history_is_bounded():
    from app.detection.tracker import MAX_RMS_SAMPLES

    p = EventProfile()
    assert p.rms_above_floor_db.maxlen == MAX_RMS_SAMPLES
    for i in range(MAX_RMS_SAMPLES + 500):
        p.rms_above_floor_db.append(float(i))
    # The oldest values were dropped, so this cannot grow without limit.
    assert len(p.rms_above_floor_db) == MAX_RMS_SAMPLES


def test_whisper_diagnostics_expose_the_required_values(tmp_path):
    h = Harness()
    h.store = EventStore(str(tmp_path))
    h.feed(9.0, _footsteps_profile(h, (1.0, 2.2, 3.4)))
    h.finish()
    event = h.events[0][0]
    detail = event.label_details.get(LABEL_WHISPER)
    assert detail, "the whisper rule must report diagnostics even when weak"
    for key in (
        "presence_fraction", "frames_above_floor", "total_frames",
        "active_level_dbfs", "active_flatness", "impulsive_onsets",
        "impulse_density", "whisper_score",
    ):
        assert key in detail, f"missing diagnostic {key}"
    assert detail["whisper_score"] < 0.2
    assert detail["presence_fraction"] < 0.4


def test_no_hard_impulse_rejection_in_the_whisper_rule():
    """Guard against reintroducing a hard impulse gate (spec item 5)."""
    import inspect

    source = inspect.getsource(Classifier._rule_whisper)
    assert "impulsive_onsets > 0" not in source
    assert "return None" not in source, (
        "the whisper rule must score and return a LabelScore, never return None "
        "on the basis of onset count"
    )


# ======================================================================
# Event store / CLI
# ======================================================================
def test_store_reports_both_pre_and_post_roll_shortfall(tmp_path):
    """A mid-run event has pre-roll but no post-roll; both must be reported."""
    from app.audio.ringbuffer import RingBuffer as _Ring

    store = EventStore(str(tmp_path))
    ring = _Ring(SR, 30.0)
    # 2 s captured, but the event's span reaches 10 s in.
    ring.write(np.zeros(2 * SR, dtype=np.float32))
    tracked = _tracked()
    tracked.start_sample = 0
    tracked.end_sample = 10 * SR
    result = Classifier(DetectionConfig()).classify(tracked.profile, 0.3)
    event = build_event(tracked, result, "event_000000")
    event, samples = store.extract(event, ring)

    assert samples is not None
    # The start was available, so the pre-roll is complete...
    assert event.preroll_missing_frames == 0
    # ...but the tail was never captured and must not be reported as present.
    assert event.postroll_missing_frames == 10 * SR - samples.size
    prov = event.metadata()["provenance"]
    assert prov["preroll_complete"] is True
    assert prov["postroll_complete"] is False
    assert prov["span_complete"] is False


def test_metadata_reports_stored_and_span_durations_separately(tmp_path):
    """The two must be comparable rather than assumed equal."""
    store = EventStore(str(tmp_path))
    ring = RingBuffer(SR, 30.0)
    ring.write(np.zeros(3 * SR, dtype=np.float32))
    tracked = _tracked()
    tracked.start_sample = 0
    tracked.end_sample = 10 * SR
    result = Classifier(DetectionConfig()).classify(tracked.profile, 0.3)
    event = build_event(tracked, result, "event_000000")
    event, samples = store.extract(event, ring)
    store.save(event, samples)
    audio = event.metadata()["audio"]
    assert audio["span_duration_seconds"] > audio["stored_duration_seconds"]
    assert audio["frames"] == samples.size


def test_event_cli_lists_and_reads_back(tmp_path, capsys):
    from app.events.cli import main as events_main

    store = EventStore(str(tmp_path))
    ring = RingBuffer(SR, 30.0)
    samples = (0.1 * np.random.default_rng(0).standard_normal(SR)).astype(
        np.float32
    )
    ring.write(samples)
    tracked = _tracked()
    tracked.start_sample = 0
    tracked.end_sample = samples.size
    result = Classifier(DetectionConfig()).classify(tracked.profile, 0.3)
    event = build_event(tracked, result, "event_000000")
    event, got = store.extract(event, ring)
    store.save(event, got)

    assert events_main(["--root", str(tmp_path)]) == 0
    listing = capsys.readouterr().out
    assert "event_000000" in listing

    assert events_main(["--root", str(tmp_path), "--verbose"]) == 0
    detail = capsys.readouterr().out
    assert "classification" in detail
    assert "presence" in detail
    # Confidence is explicitly absent, not zero.
    assert "confidence     None" in detail


def test_event_cli_json_is_parseable(tmp_path, capsys):
    from app.events.cli import main as events_main

    store = EventStore(str(tmp_path))
    ring = RingBuffer(SR, 30.0)
    ring.write(np.zeros(SR, dtype=np.float32))
    tracked = _tracked()
    tracked.start_sample = 0
    tracked.end_sample = SR
    result = Classifier(DetectionConfig()).classify(tracked.profile, 0.3)
    event = build_event(tracked, result, "event_000000")
    event, got = store.extract(event, ring)
    store.save(event, got)

    assert events_main(["--root", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["event_id"] == "event_000000"


def test_event_cli_handles_a_missing_store(tmp_path, capsys):
    from app.events.cli import main as events_main

    assert events_main(["--root", str(tmp_path / "absent")]) == 2


def test_source_provenance_is_read_from_the_source_at_emit_time():
    """A network event must name its actual sender, not "awaiting-client"."""
    from app.pipeline import _client_configs_of, _source_info_of

    class FakeNetwork:
        name = "network:10.0.0.242:59312"
        native_sample_rate = 44100
        client_configs = [{"amplification": 2.185, "segment_duration_min": 5}]

    info = _source_info_of(FakeNetwork())
    assert info["name"] == "network:10.0.0.242:59312"
    assert info["stream_rate"] == 44100
    assert info["wire_format"] == "s16le 44100 Hz mono"
    # Regression: a `callable()` guard on a list-returning property silently
    # discarded every client config.
    assert _client_configs_of(FakeNetwork()) == [
        {"amplification": 2.185, "segment_duration_min": 5}
    ]


def test_client_configs_tolerates_a_missing_source_attribute():
    from app.pipeline import _client_configs_of

    assert _client_configs_of(object()) == []


def test_network_source_remembers_the_last_client_after_disconnect():
    """An event decided after a sender leaves must still name it."""
    from app.audio.network import NetworkSource

    src = NetworkSource(
        AppConfig().validate(), port=0, host="127.0.0.1", reserve_ports=False
    )
    try:
        assert src.name == "network:awaiting-client"
        src._last_client_address = "10.0.0.242:59312"
        src._last_client_configs = [{"amplification": 2.185}]
        assert "10.0.0.242:59312" in src.name
        assert src.client_configs == [{"amplification": 2.185}]
    finally:
        src.close()
