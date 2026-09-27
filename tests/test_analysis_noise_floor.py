"""Adaptive noise floor (AGENTS.md section 8, docs/AUDIO_PIPELINE.md 6)."""

from __future__ import annotations

import numpy as np
import pytest

from app.analysis.features import FeatureExtractor, SILENCE_DB
from app.analysis.frames import FrameExtractor
from app.analysis.noise_floor import AdaptiveNoiseFloor, BandFloor
from app.config import AnalysisConfig

SR = 48000


def db(amplitude: float) -> float:
    return 20.0 * np.log10(max(amplitude, 1e-10))


class Harness:
    """Feeds a synthetic level stream through features and the floor."""

    def __init__(self, config: AnalysisConfig | None = None, seed: int = 1):
        self.config = config or AnalysisConfig()
        self.frame = self.config.frame_samples_at(SR)
        self.hop = self.config.hop_samples_at(SR)
        self.features = FeatureExtractor(SR, window=self.config.window)
        self.framer = FrameExtractor(self.frame, self.hop, None)
        self.floor = AdaptiveNoiseFloor(
            self.config, frame_rate=SR / self.hop
        )
        self.rng = np.random.default_rng(seed)
        self.states = []

    def feed(self, seconds: float, amplitude: float):
        """Feed ``seconds`` of noise at ``amplitude``, one hop-sized block at a time."""
        blocks = max(1, int(round(seconds * SR / self.hop)))
        for _ in range(blocks):
            block = (amplitude * self.rng.standard_normal(self.hop)).astype(
                np.float32
            )
            for frame in self.framer.push(block):
                self.states.append(
                    self.floor.update(self.features.analyse(frame))
                )
        return self.states

    @property
    def last(self):
        return self.states[-1]


@pytest.fixture
def harness() -> Harness:
    return Harness()


# ----------------------------------------------------------------------
# Convergence
# ----------------------------------------------------------------------
def test_floor_converges_to_the_actual_noise_level(harness):
    """It must measure the room, not assume a fixed dBFS value."""
    amplitude = 0.003  # about -50 dBFS
    harness.feed(8.0, amplitude)
    assert harness.last.initialized
    assert harness.last.overall_floor_db == pytest.approx(
        db(amplitude), abs=2.5
    )


def test_two_very_different_rooms_give_two_different_floors():
    quiet = Harness(seed=2)
    quiet.feed(8.0, 0.0005)
    loud = Harness(seed=3)
    loud.feed(8.0, 0.05)
    # A fixed threshold would report the same value for both.
    assert quiet.last.overall_floor_db < loud.last.overall_floor_db - 30
    assert quiet.last.overall_floor_db == pytest.approx(db(0.0005), abs=2.5)
    assert loud.last.overall_floor_db == pytest.approx(db(0.05), abs=2.5)


def test_silence_does_not_drift(harness):
    """A silent room must not walk the floor upward over time."""
    harness.feed(10.0, 0.0)
    early = harness.states[len(harness.states) // 2].overall_floor_db
    late = harness.last.overall_floor_db
    assert late == pytest.approx(early, abs=1.0)
    assert late == pytest.approx(SILENCE_DB, abs=0.01)


# ----------------------------------------------------------------------
# Asymmetric behaviour - the central requirement
# ----------------------------------------------------------------------
def test_a_loud_event_does_not_move_the_floor(harness):
    """A sudden sound must not become the new noise floor.

    The floor is allowed to creep only after the event has lasted longer than
    ``floor_sustain_sec``; until then it must hold.  A 40 dB event must never
    be absorbed, which is the whole point.
    """
    harness.feed(8.0, 0.003)
    before = harness.last.overall_floor_db
    mark = len(harness.states)
    harness.feed(2.0, 0.3)  # 40 dB above the floor, within the sustain window
    early = harness.states[mark + 20].overall_floor_db
    assert early == pytest.approx(before, abs=0.6)
    harness.feed(3.0, 0.3)
    after = harness.last.overall_floor_db
    # Absorbed by at most a small fraction of the event's level.
    assert after - before < 6.0, f"floor followed the event by {after - before:.1f} dB"


def test_snr_reports_the_event_while_the_floor_holds(harness):
    harness.feed(8.0, 0.003)
    harness.feed(4.0, 0.3)
    assert harness.last.overall_snr_db == pytest.approx(40.0, abs=6.0)


def test_a_sustained_increase_eventually_becomes_the_floor():
    """A genuinely noisier room must be learned, or the tool goes deaf in it."""
    harness = Harness()
    harness.feed(8.0, 0.003)
    before = harness.last.overall_floor_db
    harness.feed(25.0, 0.3)
    after = harness.last.overall_floor_db
    assert after > before + 8.0, "a persistent 40 dB increase was never learned"


def test_the_floor_falls_back_quickly(harness):
    harness.feed(6.0, 0.05)
    loud_floor = harness.last.overall_floor_db
    harness.feed(6.0, 0.002)
    assert harness.last.overall_floor_db < loud_floor - 20.0
    assert harness.last.overall_floor_db == pytest.approx(db(0.002), abs=3.0)


def test_transient_detected_is_reported_during_an_event(harness):
    """Per-frame, so a consumer sees it on every frame of the event."""
    harness.feed(8.0, 0.003)
    assert not harness.last.transient_detected
    mark = len(harness.states)
    harness.feed(2.0, 0.3)
    event_states = harness.states[mark:]
    # Every frame of the event reports it...
    assert all(s.transient_detected for s in event_states)
    # ...and the sub-window gate fired on some of them, which is the frame on
    # which a sub-window closed.
    assert any(s.gate_rejected for s in event_states)


def test_transient_gate_reports_rejections(harness):
    harness.feed(8.0, 0.003)
    harness.feed(4.0, 0.3)
    assert harness.floor.rejections > 0


# ----------------------------------------------------------------------
# Per-band behaviour
# ----------------------------------------------------------------------
def test_every_band_gets_a_floor():
    harness = Harness()
    harness.feed(8.0, 0.002)
    labels = [b.label for b in harness.last.bands]
    assert len(labels) == 9
    for band in harness.last.bands:
        assert isinstance(band, BandFloor)
        assert math_is_finite(band.floor_db)
        # A band floor is a real level, somewhere below full scale.
        assert SILENCE_DB <= band.floor_db <= 0.0


def test_band_snr_is_zero_in_a_steady_noise():
    harness = Harness()
    harness.feed(12.0, 0.002)
    quiet = [b.snr_db for b in harness.last.bands]
    assert max(quiet) < 12.0


def test_a_tone_raises_snr_only_in_its_own_band():
    harness = Harness()
    harness.feed(10.0, 0.002)
    before = {b.label: b.snr_db for b in harness.last.bands}
    # A 1 kHz tone for 2 s.
    block_hop = harness.hop
    tone = (0.2 * np.sin(2 * np.pi * 1000 * np.arange(block_hop) / SR)).astype(
        np.float32
    )
    for _ in range(200):
        for frame in harness.framer.push(tone):
            harness.states.append(
                harness.floor.update(harness.features.analyse(frame))
            )
    after = {b.label: b.snr_db for b in harness.last.bands}
    # The affected band must have risen; a far band must not have.
    assert after["1000-2000"] > before["1000-2000"] + 15.0
    assert after["20-80"] < before["20-80"] + 8.0


# ----------------------------------------------------------------------
# Plumbing
# ----------------------------------------------------------------------
def test_reset_clears_the_estimate():
    harness = Harness()
    harness.feed(8.0, 0.01)
    assert harness.floor.is_initialized
    harness.floor.reset()
    assert not harness.floor.is_initialized
    assert harness.floor.rejections == 0, "reset must clear the rejection count"


def test_state_serialises():
    import json

    harness = Harness()
    harness.feed(6.0, 0.002)
    text = json.dumps(harness.last.to_dict())
    assert "overall_floor_db" in text
    assert "snr_db" in text


def test_band_lookup_raises_for_an_unknown_label():
    harness = Harness()
    harness.feed(4.0, 0.002)
    with pytest.raises(KeyError):
        harness.last.band("nope")


def test_snr_shortcut_matches_the_state():
    harness = Harness()
    harness.feed(4.0, 0.002)
    assert harness.last.snr_db == harness.last.overall_snr_db


def test_bad_frame_rate_is_rejected():
    with pytest.raises(ValueError):
        AdaptiveNoiseFloor(AnalysisConfig(), frame_rate=0.0)


def test_a_very_quiet_room_reports_its_own_low_floor():
    """The floor must follow the measurement, not a clamp.

    Regression test: an earlier design clamped the floor to a ceiling, which
    pinned loud bands to that ceiling and reported a permanent phantom SNR in
    ordinary noise.
    """
    harness = Harness(seed=5)
    harness.feed(10.0, 0.0001)  # -80 dBFS
    assert harness.last.overall_floor_db == pytest.approx(db(0.0001), abs=3.0)


def test_band_floors_track_their_own_noise(harness):
    """Per-band floors must not be pinned to a common value."""
    harness.feed(10.0, 0.002)
    floors = {b.label: b.floor_db for b in harness.last.bands}
    assert len(set(round(v, 3) for v in floors.values())) > 1, floors


def test_fall_rate_must_be_at_least_the_rise_rate():
    """A floor that falls slower than it rises would ignore a quieting room."""
    with pytest.raises(Exception):
        AnalysisConfig(floor_rise_db_per_sec=10.0, floor_fall_db_per_sec=1.0)


def math_is_finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")
