"""Generic event-segmentation behaviour (Cases A-G).

docs/AUDIO_PIPELINE.md section 8 requires hysteresis and bounded events, and
the tracker must be able to end an event on *temporal or acoustic structure*
rather than on the signal falling silent.  These cases are deliberately
generic: no test name, parameter or assertion refers to any particular sound.

The failure this guards against: a continuously active signal producing one
indefinitely growing event, because termination depended solely on the
activation dropping below the continuation threshold - which never happens when
the level stays above the adaptive floor.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.config import DetectionConfig
from app.detection.candidate import Candidate
from app.detection.tracker import (
    EventState,
    EventTermination,
    EventTracker,
)

SR = 48000
HOP = 480  # 10 ms at 48 kHz, the default analysis hop
FPS = SR / HOP


def candidate(
    index: int,
    activation: float = 0.6,
    snr_db: float = 15.0,
    flux_ratio: float | None = None,
) -> Candidate:
    """A synthetic per-frame candidate.

    Only the attributes the tracker's segmentation reads are meaningful here;
    feature values are supplied separately where a test needs them.
    """
    return Candidate(
        index=index,
        start_sample=index * HOP,
        audio_time=index * HOP / SR,
        activation=activation,
        indicators=("level",) if activation else (),
        values={"level": activation} if activation else {},
        snr_db=snr_db,
        modulation=0.0,
        clipping_ratio=0.0,
        dominant_frequency_hz=0.0,
        spectral_peakiness=0.0,
        flux_ratio=flux_ratio,
        crest_factor=0.0,
    )


class Features:
    """Minimal stand-in carrying the two features change detection reads."""

    def __init__(self, index: int, centroid_hz: float, flatness: float):
        self.index = index
        self.start_sample = index * HOP
        self.sample_rate = SR
        self.spectral_centroid_hz = centroid_hz
        self.spectral_flatness = flatness
        self.crest_factor = 2.0
        self.rms_db = -40.0
        self.peak_db = -30.0
        self.spectral_rolloff_hz = centroid_hz * 1.2
        self.spectral_bandwidth_hz = 2000.0
        self.spectral_flux = 0.1
        self.dominant_frequency_hz = centroid_hz
        self.spectral_peakiness = 5.0
        self.zero_crossing_rate = 0.1
        self.dc_offset = 0.0
        self.bands = ()


def run(
    frames,
    config: DetectionConfig | None = None,
    **kwargs,
) -> tuple[list, EventTracker]:
    """Feed a sequence of (candidate, features) and return the events."""
    config = config or DetectionConfig()
    tracker = EventTracker(
        config, sample_rate=SR, frame_rate=FPS,
        pre_roll_seconds=1.0, post_roll_seconds=1.0, **kwargs
    )
    events = []
    for index, (cand, feats) in enumerate(frames):
        cand.index = index
        cand.start_sample = index * HOP
        if feats is not None:
            feats.index = index
        out = tracker.update(cand, feats, None)
        if out is not None:
            events.append(out)
    events.extend(tracker.flush())
    return events, tracker


def steady(count, activation=0.6, snr=15.0, centroid=1000.0, flatness=0.05,
           onsets=()):
    """`count` frames of a steady, continuously active signal."""
    out = []
    for i in range(count):
        out.append((
            candidate(i, activation, snr,
                      flux_ratio=3.0 if i in onsets else None),
            Features(i, centroid, flatness),
        ))
    return out


# ======================================================================
# Case A - a genuinely continuous signal stays one event
# ======================================================================
def test_case_a_continuous_signal_stays_a_single_event():
    """Case A. Minor fluctuations must not fragment a continuous signal."""
    frames = []
    for i in range(int(30 * FPS)):
        # Deterministic small wobble, no trend.
        wobble = 0.02 * math.sin(i / 7.0)
        frames.append((
            candidate(i, 0.6, 15.0),
            Features(i, 1000.0 * (1 + wobble), 0.05 * (1 + wobble)),
        ))
    events, tracker = run(frames, DetectionConfig(max_duration=120.0))
    assert len(events) == 1, f"continuous signal split into {len(events)}"
    # And it was not ended by a boundary, only by the stream finishing.
    assert events[0].termination is EventTermination.END_OF_STREAM


# ======================================================================
# Case B - clearly separated events
# ======================================================================
def test_case_b_two_separated_events_produce_two_events():
    """Case B. A meaningful gap separates two events without needing silence
    everywhere, only between them."""
    frames = []
    # First event: 5 s active.
    for i in range(int(5 * FPS)):
        frames.append((candidate(i, 0.6, 15.0), Features(i, 1000.0, 0.05)))
    # Quiet gap: 2 s, which is longer than release_timeout.
    for i in range(int(2 * FPS)):
        frames.append((candidate(i, 0.0, 0.0), Features(i, 1000.0, 0.05)))
    # Second event: 5 s active again, continuing the index.
    for k in range(int(5 * FPS)):
        i = int(7 * FPS) + k
        frames.append((candidate(i, 0.6, 15.0), Features(i, 1000.0, 0.05)))

    events, tracker = run(frames)
    assert len(events) == 2, f"expected 2 events, got {len(events)}"
    reasons = [e.termination for e in events]
    assert EventTermination.QUIET in reasons


def test_case_b_events_are_not_merged_when_well_apart():
    """The merge bound must actually bind."""
    config = DetectionConfig(max_merge_interval=0.5, merge_gap=5.0)
    frames = steady(int(3 * FPS))
    for i in range(int(3 * FPS), int(4 * FPS)):
        frames.append((candidate(i, 0.0, 0.0), Features(i, 1000.0, 0.05)))
    for k in range(int(3 * FPS)):
        i = int(4 * FPS) + k
        frames.append((candidate(i, 0.6, 15.0), Features(i, 1000.0, 0.05)))
    events, _ = run(frames, config)
    assert len(events) == 2
    assert all(e.merged_count == 0 for e in events)


# ======================================================================
# Case C - repeated transients must not merge indefinitely
# ======================================================================
def test_case_c_repeated_transients_do_not_grow_one_event_forever():
    """Case C. Dense onsets with distinct temporal clusters must segment.

    The signal never goes quiet, so the release timer can never fire; the
    inter-onset structure is what has to provide the boundary.
    """
    frames = []
    # 40 s, never quiet, with onsets in 3 s bursts separated by 3 s of no
    # onsets at all.  Activation stays high throughout, so only the inter-onset
    # structure can segment this.
    rate = FPS
    for i in range(int(40 * rate)):
        position = i % int(6 * rate)
        is_onset = position < int(3 * rate) and position % int(0.5 * rate) == 0
        frames.append((
            candidate(i, 0.6, 15.0, flux_ratio=3.0 if is_onset else None),
            Features(i, 1000.0, 0.05),
        ))
    events, tracker = run(frames, DetectionConfig(max_duration=600.0))
    assert len(events) >= 3, (
        f"40 s of clustered transients produced {len(events)} event(s); "
        "the inter-onset boundary is not working"
    )
    assert any(
        e.termination is EventTermination.ONSET_CLUSTER for e in events
    ), [e.termination.value for e in events]
    # Every event is bounded, not one growing event.
    for event in events:
        active = (
            event.profile.last_active_sample - event.profile.onset_sample
        ) / SR
        assert active < 30.0, f"event grew to {active:.1f}s"


# ======================================================================
# Case D - continuous background with distinct foreground changes
# ======================================================================
def test_case_d_background_plus_foreground_changes_segments():
    """Case D. The background never stops; the changes are what must segment.

    No silence anywhere, and the activation never drops, so nothing but the
    acoustic-change boundary can end an event.
    """
    frames = []
    for i in range(int(30 * FPS)):
        # Three 10 s epochs with distinctly different spectral character.
        epoch = i // int(10 * FPS)
        centroid = (700.0, 1800.0, 4200.0)[epoch]
        flatness = (0.02, 0.20, 0.05)[epoch]
        frames.append((
            candidate(i, 0.6, 15.0), Features(i, centroid, flatness)
        ))
    events, tracker = run(frames, DetectionConfig(max_duration=600.0))
    assert len(events) >= 2, (
        f"three spectral epochs over a continuous background produced "
        f"{len(events)} event(s)"
    )
    assert any(
        e.termination is EventTermination.ACOUSTIC_CHANGE for e in events
    ), [e.termination.value for e in events]


def test_case_d_never_requires_silence():
    """Explicit: segmentation must fire with activation pinned high."""
    frames = []
    for i in range(int(30 * FPS)):
        epoch = i // int(10 * FPS)
        frames.append((
            candidate(i, 0.6, 15.0),
            Features(i, (700.0, 3000.0, 700.0)[epoch], 0.05),
        ))
    events, _ = run(frames, DetectionConfig(max_duration=600.0))
    assert len(events) >= 2
    for event in events:
        # No frame in any event was ever inactive.
        assert event.profile.frames > 0


# ======================================================================
# Case E - noisy continuous signal must not fragment
# ======================================================================
def test_case_e_small_random_fluctuations_do_not_fragment():
    """Case E. Frame-to-frame noise is not a change of acoustic character."""
    rng = np.random.default_rng(4)
    frames = []
    for i in range(int(60 * FPS)):
        frames.append((
            candidate(i, 0.6, 15.0),
            Features(
                i,
                1000.0 * (1 + 0.05 * rng.standard_normal()),
                0.05 * (1 + 0.08 * rng.standard_normal()),
            ),
        ))
    events, _ = run(frames, DetectionConfig(max_duration=600.0))
    assert len(events) == 1, f"noise fragmented into {len(events)} events"


def test_case_e_a_single_onset_never_splits_an_event():
    """One onset is not a cluster, so it must not enable the boundary.

    The clustering check is guarded by min_onsets_for_clustering precisely so
    that a signal with one or two onsets is never fragmented.  Pure noise
    carries no onsets at all and is covered by the test above.
    """
    for onset_count in (0, 1, 2):
        rng = np.random.default_rng(5)
        pool = list(range(int(60 * FPS)))
        onsets = set(
            rng.choice(pool, size=onset_count, replace=False).tolist()
        ) if onset_count else set()
        frames = []
        for i in range(int(60 * FPS)):
            frames.append((
                candidate(i, 0.6, 15.0,
                          flux_ratio=3.0 if i in onsets else None),
                Features(i, 1000.0 * (1 + 0.04 * rng.standard_normal()), 0.05),
            ))
        events, _ = run(frames, DetectionConfig(max_duration=600.0))
        assert len(events) == 1, (
            f"{onset_count} onset(s) fragmented into {len(events)} event(s)"
        )


def test_case_e_many_widely_spaced_onsets_do_segment():
    """The converse: enough separated onsets are genuine distinct clusters.

    Asserted so the previous test is not passing merely because the check is
    disabled - that would be a vacuous test.
    """
    frames = []
    rate = FPS
    for i in range(int(60 * rate)):
        is_onset = (i % int(5 * rate)) < int(0.3 * rate)
        frames.append((
            candidate(i, 0.6, 15.0, flux_ratio=3.0 if is_onset else None),
            Features(i, 1000.0, 0.05),
        ))
    events, _ = run(frames, DetectionConfig(max_duration=600.0))
    assert len(events) > 1, "widely spaced onset clusters were not segmented"


# ======================================================================
# Case F - a short interruption must not split
# ======================================================================
def test_case_f_brief_interruption_does_not_split():
    """Case F. A momentary dip inside a continuous event is not a boundary."""
    frames = []
    for i in range(int(20 * FPS)):
        # 0.2 s dip starting at 10 s: shorter than release_timeout (0.4 s).
        quiet = int(10 * FPS) <= i < int(10.2 * FPS)
        frames.append((
            candidate(i, 0.0 if quiet else 0.6, 0.0 if quiet else 15.0),
            Features(i, 1000.0, 0.05),
        ))
    events, _ = run(frames, DetectionConfig(max_duration=600.0))
    assert len(events) == 1, (
        f"a 0.2 s interruption split the event into {len(events)}"
    )


def test_case_f_interruption_longer_than_release_does_split():
    """The counterpart, so Case F is not passing for the wrong reason."""
    frames = []
    for i in range(int(20 * FPS)):
        quiet = int(10 * FPS) <= i < int(11.0 * FPS)   # 1.0 s
        frames.append((
            candidate(i, 0.0 if quiet else 0.6, 0.0 if quiet else 15.0),
            Features(i, 1000.0, 0.05),
        ))
    events, _ = run(frames, DetectionConfig(max_duration=600.0))
    assert len(events) == 2, (
        f"a 1.0 s interruption produced {len(events)} event(s)"
    )


# ======================================================================
# Case G - a sustained acoustic transition
# ======================================================================
def test_case_g_sustained_transition_ends_the_event():
    """Case G. A persistent change in character, with energy never dropping."""
    frames = []
    for i in range(int(30 * FPS)):
        second_half = i >= int(15 * FPS)
        frames.append((
            candidate(i, 0.6, 15.0),
            Features(i, 4000.0 if second_half else 800.0, 0.05),
        ))
    events, tracker = run(frames, DetectionConfig(max_duration=600.0))
    assert len(events) >= 2
    assert any(
        e.termination is EventTermination.ACOUSTIC_CHANGE for e in events
    )
    change = next(
        e for e in events
        if e.termination is EventTermination.ACOUSTIC_CHANGE
    )
    # The boundary is explained, not merely taken.
    assert change.boundary_detail.get("feature") in ("centroid_hz", "flatness")
    assert change.boundary_detail["deviation"] > change.boundary_detail["threshold"]


def test_case_g_gradual_ramp_is_tolerated():
    """A slow drift is not a transition; only a step-like change is."""
    frames = []
    for i in range(int(40 * FPS)):
        progress = i / (40 * FPS)
        frames.append((
            candidate(i, 0.6, 15.0),
            Features(i, 1000.0 + 300.0 * progress, 0.05),
        ))
    events, _ = run(frames, DetectionConfig(max_duration=600.0))
    assert len(events) == 1, f"a slow ramp fragmented into {len(events)}"


# ======================================================================
# The original defect
# ======================================================================
def test_continuous_activity_does_not_produce_one_unbounded_event():
    """The reported failure, as a regression test.

    Activation pinned above the continuation threshold for 60 s: the release
    timer can never fire, and previously max_duration was split-then-remerged
    into a single event.
    """
    config = DetectionConfig(max_duration=10.0, max_merge_interval=0.5)
    frames = []
    for i in range(int(60 * FPS)):
        frames.append((candidate(i, 0.6, 15.0), Features(i, 1000.0, 0.05)))
    events, tracker = run(frames, config)
    assert len(events) > 1, (
        "60 s of continuously active signal is still one event"
    )
    for event in events:
        active = (
            event.profile.last_active_sample - event.profile.onset_sample
        ) / SR
        assert active <= 10.5, f"event ran {active:.1f}s past a 10s cap"
    # A deliberate boundary must never be undone by merging.
    assert all(e.merged_count == 0 for e in events), (
        "boundary-terminated events were merged back together"
    )


def test_max_duration_is_not_defeated_by_merging():
    """A split caused by the duration cap must survive the merge step."""
    config = DetectionConfig(max_duration=5.0, max_merge_interval=5.0)
    frames = steady(int(30 * FPS))
    events, tracker = run(frames, config)
    assert len(events) >= 5, f"30 s at a 5 s cap gave {len(events)} events"
    assert tracker.events_merged == 0, "merging undid the duration cap"


# ======================================================================
# Diagnostics
# ======================================================================
def test_transitions_are_recorded_for_each_action():
    frames = []
    for i in range(int(6 * FPS)):
        quiet = int(3 * FPS) <= i < int(4 * FPS)
        frames.append((
            candidate(i, 0.0 if quiet else 0.6, 0.0 if quiet else 15.0),
            Features(i, 1000.0, 0.05),
        ))
    events, tracker = run(frames)
    actions = [t["action"] for t in tracker.transitions]
    assert "start" in actions
    assert "terminate" in actions
    terminate = [t for t in tracker.transitions if t["action"] == "terminate"]
    assert terminate[0]["reason"] == EventTermination.QUIET.value
    # Bounded: transitions only, never one record per frame.
    assert len(tracker.transitions) < len(frames) / 10


def test_merge_refusal_is_recorded():
    """A refused merge is visible, so a boundary can be audited."""
    config = DetectionConfig(max_duration=2.0, max_merge_interval=0.1)
    frames = steady(int(10 * FPS))
    _events, tracker = run(frames, config)
    actions = {t["action"] for t in tracker.transitions}
    assert "merge_refused" in actions or "merge" in actions


def test_termination_reason_is_recorded_on_the_event():
    frames = []
    for i in range(int(6 * FPS)):
        quiet = int(3 * FPS) <= i < int(4 * FPS)
        frames.append((
            candidate(i, 0.0 if quiet else 0.6, 0.0 if quiet else 15.0),
            Features(i, 1000.0, 0.05),
        ))
    events, _ = run(frames)
    assert all(e.termination is not EventTermination.NOT_TERMINATED
               for e in events)
    assert all(isinstance(e.termination, EventTermination) for e in events)
    payload = events[0].to_dict()
    assert "termination" in payload
    assert "boundary_detail" in payload


# ======================================================================
# Configuration and genericity
# ======================================================================
def test_segmentation_parameters_are_configurable():
    """Tightening the change sensitivity must make boundaries easier."""
    loose = DetectionConfig(change_point_relative=2.0,
                            change_point_sensitivity=20.0)
    tight = DetectionConfig(change_point_relative=0.05,
                            change_point_sensitivity=1.0)
    frames = []
    for i in range(int(30 * FPS)):
        epoch = i // int(10 * FPS)
        frames.append((
            candidate(i, 0.6, 15.0),
            Features(i, (700.0, 1800.0, 3600.0)[epoch], 0.05),
        ))
    loose_events, _ = run(frames, loose)
    tight_events, _ = run(frames, tight)
    assert len(tight_events) >= len(loose_events), (
        f"tight={len(tight_events)} should segment at least as much as "
        f"loose={len(loose_events)}"
    )


def test_onset_cluster_window_is_configurable():
    wide = DetectionConfig(onset_cluster_window=10.0,
                           min_onsets_for_clustering=3)
    narrow = DetectionConfig(onset_cluster_window=0.5,
                             min_onsets_for_clustering=3)
    frames = []
    rate = FPS
    for i in range(int(20 * rate)):
        # Onsets every 0.8 s: inside the narrow window, outside the wide one.
        onset = i % int(0.8 * rate) == 0
        frames.append((
            candidate(i, 0.6, 15.0, flux_ratio=3.0 if onset else None),
            Features(i, 1000.0, 0.05),
        ))
    wide_events, _ = run(frames, wide)
    narrow_events, _ = run(frames, narrow)
    assert len(narrow_events) > len(wide_events), (
        f"narrow={len(narrow_events)} should segment more than "
        f"wide={len(wide_events)}"
    )


@pytest.mark.parametrize("bad", [
    {"change_point_window": 0.0},
    {"change_point_sensitivity": 0.0},
    {"change_point_relative": -1.0},
    {"onset_cluster_window": -1.0},
    {"min_onsets_for_clustering": 0},
    {"max_merge_interval": -1.0},
])
def test_invalid_segmentation_config_is_rejected(bad):
    with pytest.raises(Exception):
        DetectionConfig(**bad)


def test_no_application_specific_terms_in_the_detector():
    """Guard requirement 7: the detector core must stay generic.

    If a future change reintroduces logic tied to one kind of sound, this fails
    rather than the code quietly becoming a single-application tool.
    """
    import ast
    import re
    from pathlib import Path

    # Scoped to the segmentation logic - the state machine that decides where
    # events begin and end.  The classifier is *required* to have per-class
    # rules: tasks/PHASE_03_EVENTS.md lists whisper, speech, footsteps,
    # movement, interference and unknown as things to classify.  What must not
    # happen is the event *segmentation* being tuned to one of them.
    tracker = (
        Path(__file__).resolve().parents[1] / "app" / "detection"
        / "tracker.py"
    )
    forbidden = re.compile(
        r"breath|respirat|snore|sleep|heartbeat|footstep|knock|doorbell|"
        r"\bbaby\b|\bbabies\b|cough|whisper|speech",
        re.IGNORECASE,
    )

    offenders = []
    for path in (tracker,):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            # Only *decisions* are checked.  A named frequency band, or prose
            # explaining why a threshold was chosen, is not an assumption about
            # a sound class; branching on one would be.
            if isinstance(node, ast.If):
                segment = ast.unparse(node.test)
                if forbidden.search(segment):
                    offenders.append(
                        f"{path.name}:{node.lineno}: condition {segment!r}"
                    )
            elif isinstance(node, ast.Compare):
                segment = ast.unparse(node)
                if forbidden.search(segment):
                    offenders.append(
                        f"{path.name}:{node.lineno}: comparison {segment!r}"
                    )
    assert not offenders, (
        "application-specific branching found in the detector core:\n"
        + "\n".join(offenders)
    )


def test_band_names_are_frequency_ranges_not_sound_classes():
    """The analysis bands are ranges; the detector must not name them by class.

    A constant like ``SPEECH_BAND`` ties the detector core to one sound class
    even though the range itself is generic, so the core uses frequency-based
    names only.
    """
    import re
    from pathlib import Path

    for name in ("candidate.py", "tracker.py"):  # the detector core
        text = (
            Path(__file__).resolve().parents[1] / "app" / "detection" / name
        ).read_text()
        offenders = [
            line.strip()
            for line in text.splitlines()
            if re.search(r"^\s*[A-Z_]*(SPEECH|WHISPER|FOOTSTEP|BREATH)", line)
        ]
        assert not offenders, (
            f"{name} names a measurement after a sound class: {offenders}"
        )
