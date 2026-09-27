"""Event state machine.

docs/AUDIO_PIPELINE.md section 8: a bare threshold produces

    candidate / no candidate / candidate / no candidate

for every frame of one real event.  This is the hysteresis that prevents it:

* **onset threshold** - a higher bar to start an event,
* **continuation threshold** - a lower bar to keep it going,
* **release timeout** - how long the signal must stay quiet before the event
  actually ends, which is what captures trailing sounds (section 10),
* **minimum duration** - rejects blips that were never events,
* **maximum duration** - splits an event that has clearly ended or is running
  away,
* **merge gap** - folds two detections that are really one thing.

Pre-roll and post-roll are applied to the *stored span*, not to the detection
decision: an event starting at T is stored from ``T - pre_roll``
(section 9), and ends ``post_roll`` after the last active frame (section 10).
The pre-roll can only come from the ring buffer, which is why the buffer exists;
whether it was actually available is recorded rather than assumed.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from ..analysis.features import FrameFeatures
from ..analysis.noise_floor import NoiseFloorState
from ..config import DetectionConfig
from ..events.fingerprint import MAX_LEVEL_SERIES, RunningMoments
from .candidate import (
    HIGH_BAND,
    LOW_BAND,
    MID_BAND,
    Candidate,
    CandidateDetector,
)


#: Upper bound on the per-frame RMS values retained per event for the
#: "active level" measurement.  At the default 100 analysis frames per second
#: this is over three minutes of above-floor audio, which comfortably covers any
#: single event, and it keeps per-frame accumulation bounded and predictable
#: (docs/PERFORMANCE.md section 7).  The deque drops the oldest samples, so the
#: measurement degrades to "the most recent above-floor audio" rather than
#: growing without limit.
MAX_RMS_SAMPLES = 20_000


class EventState(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    RELEASING = "releasing"


class EventTermination(Enum):
    """Why an event ended.

    Recorded so a boundary can be explained rather than guessed at.  The
    distinction that matters operationally is ``QUIET`` (activity genuinely
    stopped) versus the change-based reasons, which fire while the signal is
    still fully active.
    """

    QUIET = "quiet"
    ACOUSTIC_CHANGE = "acoustic_change"
    ONSET_CLUSTER = "onset_cluster"
    MAX_DURATION = "max_duration"
    END_OF_STREAM = "end_of_stream"
    NOT_TERMINATED = "not_terminated"


#: Terminations that represent a deliberate boundary.  An event that ended
#: because of one of these must not be merged into its neighbour, or the
#: boundary would simply be undone - which is exactly what defeated
#: max_duration before.
_BOUNDARY_TERMINATIONS = frozenset({
    EventTermination.ACOUSTIC_CHANGE,
    EventTermination.ONSET_CLUSTER,
    EventTermination.MAX_DURATION,
})


@dataclass
class EventProfile:
    """Measurements accumulated over one event, for the classifier.

    Everything here is a real aggregate of the frames that belonged to the
    event.  Nothing is invented, and nothing is a confidence value.
    """

    # Timing
    onset_sample: int = 0
    last_active_sample: int = 0
    first_candidate_sample: int = 0

    # Level
    peak_snr_db: float = 0.0
    mean_snr_db: float = 0.0
    peak_dbfs: float = -200.0
    mean_rms_dbfs: float = -200.0
    peak_level_dbfs: float = -200.0

    # Shape
    max_crest: float = 0.0
    mean_crest: float = 0.0
    mean_modulation: float = 0.0
    max_modulation: float = 0.0
    max_clipping_ratio: float = 0.0
    max_peakiness: float = 0.0
    mean_peakiness: float = 0.0
    mean_flatness: float = 0.0
    mean_centroid_hz: float = 0.0
    centroid_at_peak_hz: float = 0.0
    mean_bandwidth_hz: float = 0.0
    max_bandwidth_hz: float = 0.0
    max_flux: float = 0.0

    # Bands (mean SNR over the event, against each band's own floor)
    mean_low_band_snr_db: float = 0.0
    max_low_band_snr_db: float = 0.0
    mean_high_band_snr_db: float = 0.0
    max_high_band_snr_db: float = 0.0
    mean_mid_band_snr_db: float = 0.0
    # Ratio of high to low band activity; a whisper leans high, a footstep low.
    high_to_low_ratio_db: float = 0.0

    # Dominant frequency, and how stable it was
    dominant_frequency_hz: float = 0.0
    dominant_frequency_spread_hz: float = 0.0

    # Impulsive onsets and their spacing, for footstep sequences
    impulsive_onsets: int = 0
    onset_intervals: list[float] = field(default_factory=list)
    last_onset_sample: int = 0

    # Bookkeeping
    frames: int = 0
    # Frames for which analysis features were supplied.  Several accumulators
    # are only updated on those frames, so they must be averaged by this count.
    frames_with_features: int = 0
    active_frames: int = 0
    candidate_frames: int = 0
    # ------------------------------------------------------------------
    # Presence and above-floor measurements.
    #
    # An event's *mean* level and *mean* spectrum are diluted by the silence
    # between its parts: three footsteps in a quiet room have a low mean RMS and
    # a high mean flatness, which is exactly the profile of a sustained whisper.
    # Everything below is therefore measured only over frames that were
    # genuinely above the adaptive floor, so the question answered is "how loud
    # and what colour is this sound *when it is actually present*".
    #
    # All of it is relative to the adaptive noise floor, never to an absolute
    # dBFS value (AGENTS.md section 8).
    # ------------------------------------------------------------------

    # Frames whose level was meaningfully above the adaptive floor.
    frames_above_floor: int = 0
    # ------------------------------------------------------------------
    # Compact per-event statistics, accumulated in O(1) space and time from
    # values the detector has already computed.  These exist so a fingerprint
    # can be built without retaining every frame: moments give mean, spread,
    # and trend for a handful of scalars instead of a full feature series.
    # ------------------------------------------------------------------
    level_moments: RunningMoments = field(default_factory=RunningMoments)
    snr_moments: RunningMoments = field(default_factory=RunningMoments)
    centroid_moments: RunningMoments = field(default_factory=RunningMoments)
    flatness_moments: RunningMoments = field(default_factory=RunningMoments)
    # Bounded, decimated level series for the coarse envelope.  Bounded so a
    # long event cannot grow without limit; not every frame for a long event.
    level_series: deque = field(
        default_factory=lambda: deque(maxlen=MAX_LEVEL_SERIES)
    )
    # Per-frame RMS of those above-floor frames, in dBFS, so the median and a
    # high percentile are available; a mean over all frames is not
    # representative.  Bounded: the deque keeps the most recent
    # MAX_RMS_SAMPLES values, so a long event cannot grow this without limit.
    rms_above_floor_db: deque = field(
        default_factory=lambda: deque(maxlen=MAX_RMS_SAMPLES)
    )
    # Spectral flatness summed over above-floor frames only, for the same
    # reason: gap frames would otherwise make every impulsive event look
    # broadband.
    flatness_above_sum: float = 0.0
    flatness_above_count: int = 0
    # Per-frame centroid of above-floor frames, and their peakiness, so the
    # spectral evidence is also not gap-dominated.
    centroid_above_sum: float = 0.0
    peakiness_above_max: float = 0.0
    # Analysis rate this event was measured at, so time-normalised measures do
    # not silently assume the default hop.
    frames_per_second: float = 0.0
    indicators: dict[str, int] = field(default_factory=dict)
    indicator_activation: dict[str, float] = field(default_factory=dict)
    truncated: bool = False
    gapped: bool = False
    release_frames: int = 0

    @property
    def is_impulsive(self) -> bool:
        """A sharp transient, evidenced by a crest factor above the noise.

        Note this is *not* true of a footstep; see :attr:`impulsive_onsets`.
        """
        return self.max_crest >= 5.0

    @property
    def total_frames(self) -> int:
        """Total analysis frames accumulated for this event."""
        return self.frames

    @property
    def presence_fraction(self) -> float:
        """Share of the event's frames that were actually above the floor.

        Clamped to [0, 1].  Zero when the event has no frames, rather than
        dividing by zero.

        This is the single most useful discriminator in the classifier: a
        sustained whisper is present in nearly every frame, while footsteps are
        present in a few bursts out of many, and both have a low mean level.
        """
        if self.frames <= 0:
            return 0.0
        return max(0.0, min(1.0, self.frames_above_floor / self.frames))

    @property
    def active_level_dbfs(self) -> float:
        """Level of the sound *while it is present*, not its event average.

        The median of the above-floor frames.  A median rather than a mean so
        that a couple of very loud frames cannot drag the figure upward, and
        over above-floor frames only so that the silence between footsteps is
        excluded entirely.

        Returns the silence floor when the event had no above-floor frame,
        which is the honest answer for an event that never rose above its
        background.
        """
        if not self.rms_above_floor_db:
            return -200.0
        return float(np.median(self.rms_above_floor_db))

    @property
    def active_level_p90_dbfs(self) -> float:
        """90th percentile of above-frame levels: a loud-but-not-peak figure."""
        if not self.rms_above_floor_db:
            return -200.0
        return float(np.percentile(self.rms_above_floor_db, 90.0))

    @property
    def active_flatness(self) -> float:
        """Mean spectral flatness over above-floor frames only."""
        if self.flatness_above_count <= 0:
            return self.mean_flatness
        return self.flatness_above_sum / self.flatness_above_count

    @property
    def active_centroid_hz(self) -> float:
        """Mean spectral centroid over above-floor frames only."""
        if self.frames_above_floor <= 0:
            return self.mean_centroid_hz
        return self.centroid_above_sum / self.frames_above_floor

    @property
    def impulse_density(self) -> float:
        """Impulsive onsets per analysis frame, i.e. ``onsets / max(frames, 1)``.

        Deliberately frame-normalised rather than time-normalised, so it does
        not depend on the hop size: a different analysis frame rate gives the
        same number for the same acoustic event.

        Used as a *gentle penalty* on the whisper score, never as a rejection.
        Whispered speech contains consonant transients - "t", "k", "p", "sh",
        "ch" all produce short impulses - so a whisper must not be made
        impossible to classify by having any onset at all.
        """
        return self.impulsive_onsets / max(self.frames, 1)

    @property
    def onset_density_per_second(self) -> float:
        """Impulsive onsets per second of event, for the repetition rules.

        Needs the real frame rate, which is recorded on the profile, so this
        stays correct if the analysis hop is reconfigured.
        """
        if self.frames <= 0 or self.frames_per_second <= 0:
            return 0.0
        seconds = self.frames / self.frames_per_second
        if seconds <= 0:
            return 0.0
        return self.impulsive_onsets / seconds

    @property
    def onset_count(self) -> int:
        """Distinct onsets in the event, from flux jumps.

        One impulse can span several frames, so this counts onsets rather than
        frames.  A sequence of several is stronger evidence of repeated events
        than a single one, which is how footsteps are told from one bang.
        """
        return self.impulsive_onsets

    @property
    def is_tonal(self) -> bool:
        return self.max_peakiness >= 100.0

    @property
    def is_clipped(self) -> bool:
        return self.max_clipping_ratio > 0.0

    @property
    def duration_seconds(self) -> float:
        return 0.0  # set by the tracker, which knows the sample rate

    def to_dict(self) -> dict:
        return {
            "peak_snr_db": self.peak_snr_db,
            "mean_snr_db": self.mean_snr_db,
            "peak_dbfs": self.peak_dbfs,
            "mean_rms_dbfs": self.mean_rms_dbfs,
            "max_crest": self.max_crest,
            "mean_crest": self.mean_crest,
            "mean_modulation": self.mean_modulation,
            "max_clipping_ratio": self.max_clipping_ratio,
            "max_peakiness": self.max_peakiness,
            "mean_flatness": self.mean_flatness,
            "mean_centroid_hz": self.mean_centroid_hz,
            "mean_bandwidth_hz": self.mean_bandwidth_hz,
            "max_flux": self.max_flux,
            "mean_low_band_snr_db": self.mean_low_band_snr_db,
            "max_low_band_snr_db": self.max_low_band_snr_db,
            "mean_high_band_snr_db": self.mean_high_band_snr_db,
            "max_high_band_snr_db": self.max_high_band_snr_db,
            "mean_mid_band_snr_db": self.mean_mid_band_snr_db,
            "high_to_low_ratio_db": self.high_to_low_ratio_db,
            "dominant_frequency_hz": self.dominant_frequency_hz,
            "dominant_frequency_spread_hz": self.dominant_frequency_spread_hz,
            "impulsive_onsets": self.impulsive_onsets,
            "onset_intervals_s": [round(v, 4) for v in self.onset_intervals],
            "frames": self.frames,
            "active_frames": self.active_frames,
            "frames_above_floor": self.frames_above_floor,
            "moments": {
                "level": self.level_moments.to_dict(),
                "snr": self.snr_moments.to_dict(),
                "centroid": self.centroid_moments.to_dict(),
                "flatness": self.flatness_moments.to_dict(),
            },
            "total_frames": self.total_frames,
            "presence_fraction": round(self.presence_fraction, 4),
            "active_level_dbfs": self.active_level_dbfs,
            "active_level_p90_dbfs": round(self.active_level_p90_dbfs, 2),
            "active_flatness": round(self.active_flatness, 4),
            "active_centroid_hz": round(self.active_centroid_hz, 1),
            "impulse_density": round(self.impulse_density, 5),
            "onset_density_per_second": round(self.onset_density_per_second, 3),
            "candidate_frames": self.candidate_frames,
            "indicators": dict(self.indicators),
            "truncated": self.truncated,
            "gapped": self.gapped,
        }


@dataclass
class TrackedEvent:
    """A detected event span, before audio extraction and classification."""

    start_sample: int
    end_sample: int
    sample_rate: int
    profile: EventProfile
    # True when the pre-roll asked for more history than the ring buffer had.
    preroll_missing_frames: int = 0
    preroll_requested_frames: int = 0
    # True when the event hit max_duration and was cut short.
    truncated: bool = False
    # Set by the tracker when two detections were merged into this one.
    merged_count: int = 0
    event_id: Optional[str] = None
    # Why this event ended.  See EventTermination.
    termination: "EventTermination" = EventTermination.NOT_TERMINATED
    # Diagnostics for the transition, so a boundary can be explained.
    boundary_detail: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return (self.end_sample - self.start_sample) / self.sample_rate

    @property
    def start_seconds(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_seconds(self) -> float:
        return self.end_sample / self.sample_rate

    def to_dict(self) -> dict:
        return {
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "sample_rate": self.sample_rate,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration_seconds": self.duration,
            "preroll_missing_frames": self.preroll_missing_frames,
            "preroll_requested_frames": self.preroll_requested_frames,
            "truncated": self.truncated,
            "merged_count": self.merged_count,
            "termination": self.termination.value,
            "boundary_detail": dict(self.boundary_detail),
            "profile": self.profile.to_dict(),
        }


class EventTracker:
    """Turns a stream of candidates into event spans.

    Feed every candidate in order.  Completed events come back from
    :meth:`update` and from :meth:`flush`.
    """

    def __init__(
        self,
        config: Optional[DetectionConfig] = None,
        sample_rate: int = 48000,
        frame_rate: float = 100.0,
        pre_roll_seconds: float = 5.0,
        post_roll_seconds: float = 5.0,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        self.config = config or DetectionConfig()
        self.sample_rate = int(sample_rate)
        self.frame_rate = float(frame_rate)
        self.hop_samples = max(1, int(round(sample_rate / frame_rate)))
        self.pre_roll_frames = int(round(pre_roll_seconds * sample_rate))
        self.post_roll_frames = int(round(post_roll_seconds * sample_rate))
        self.release_frames = max(
            1, int(round(self.config.release_timeout * frame_rate))
        )
        self.min_duration_frames = int(
            round(self.config.min_duration * frame_rate)
        )
        self.max_duration_frames = int(
            round(self.config.max_duration * frame_rate)
        )
        self.merge_gap_frames = int(round(self.config.merge_gap * frame_rate))

        self._state = EventState.IDLE
        self._profile: Optional[EventProfile] = None
        self._frame_in_event = 0
        self._release_count = 0
        self._prev_active_end: Optional[int] = None
        self._pending: Optional[TrackedEvent] = None
        # Set when a boundary was signalled, so the merge step cannot undo it.
        self._boundary_signalled = False
        # Trailing window of already-computed values, used to detect a change
        # in acoustic character.  No new analysis is performed.
        window = max(2, int(round(config.change_point_window * frame_rate)))
        self._trail: deque = deque(maxlen=window)
        # Onset sample times inside the current event, for clustering.
        self._onset_samples: list[int] = []
        self._last_onset_sample: Optional[int] = None
        # Recent state transitions, for diagnostics.
        self.transitions: list[dict] = []
        self.max_transitions = 200

        self.events_emitted = 0
        self.events_merged = 0
        self.events_discarded_short = 0
        self.events_truncated = 0

    # ------------------------------------------------------------------
    @property
    def state(self) -> EventState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state is not EventState.IDLE

    def reset(self) -> None:
        self._state = EventState.IDLE
        self._profile = None
        self._frame_in_event = 0
        self._release_count = 0
        self._prev_active_end = None
        self._pending = None
        self._boundary_signalled = False
        self._trail.clear()
        self._onset_samples.clear()
        self._last_onset_sample = None

    def _note(self, action: str, **fields) -> None:
        """Record one state transition.  Transitions only, never per frame."""
        record = {
            "action": action,
            "at_seconds": round(self._frame_in_event * self.hop_samples
                                 / self.sample_rate, 3),
            **fields,
        }
        self.transitions.append(record)
        if len(self.transitions) > self.max_transitions:
            del self.transitions[:-self.maxitions]

    # ------------------------------------------------------------------
    def update(
        self,
        candidate: Candidate,
        features: Optional[FrameFeatures] = None,
        floor: Optional[NoiseFloorState] = None,
    ) -> Optional[TrackedEvent]:
        """Feed one frame.  Returns an event when one completes.

        ``features`` and ``floor`` are needed to aggregate the profile; without
        them the event still completes but its measurements are empty.
        """
        config = self.config

        if self._state is EventState.IDLE:
            if candidate.activation >= config.onset_threshold:
                self._begin(candidate)
                self._accumulate(candidate, features, floor)
                return None
            return None

        # --- active -----------------------------------------------------
        self._accumulate(candidate, features, floor)
        assert self._profile is not None

        if candidate.activation >= config.continuation_threshold:
            self._profile.last_active_sample = candidate.start_sample
            self._state = EventState.ACTIVE
            self._release_count = 0
        else:
            self._release_count += 1
            if self._state is EventState.ACTIVE:
                self._state = EventState.RELEASING

        # A change in acoustic character, or a distinct temporal cluster of
        # onsets, can end the event while the signal is still fully active.
        # Neither requires silence: both are measured against the recent
        # acoustic state, not against zero.
        boundary = self._evaluate_boundaries(candidate, features)
        if boundary is not None:
            reason, detail = boundary
            self._boundary_signalled = True
            self._note(
                "terminate", reason=reason.value, **detail
            )
            return self._finish(reason, detail)

        if self._release_count >= self.release_frames:
            self._note("terminate", reason=EventTermination.QUIET.value,
                       quiet_frames=self._release_count)
            return self._finish(EventTermination.QUIET)
        if self._frame_in_event >= self.max_duration_frames:
            self._profile.truncated = True
            self.events_truncated += 1
            self._note("terminate", reason=EventTermination.MAX_DURATION.value,
                       frames_in_event=self._frame_in_event)
            return self._finish(EventTermination.MAX_DURATION)
        return None

    # ------------------------------------------------------------------
    def _evaluate_boundaries(self, candidate, features):
        """Decide whether a change in the signal ends the current event.

        Two independent, generic conditions.  Both compare against the recent
        acoustic state rather than against silence, so a boundary can occur
        inside a signal that never returns to the noise floor.

        Returns ``(EventTermination, detail)`` or None.
        """
        config = self.config
        # Too early to say anything meaningful; the minimum duration is a
        # guard against fragmenting the very start of an event.
        if self._frame_in_event < self.min_duration_frames:
            return None

        # --- 1. inter-onset structure -----------------------------------
        # Only meaningful once there are enough onsets to have structure.  This
        # is a guard, not a definition: no fixed number of onsets defines an
        # event, it only enables the check.
        window_samples = config.onset_cluster_window * self.sample_rate
        if (
            window_samples > 0
            and len(self._onset_samples) >= config.min_onsets_for_clustering
            and self._last_onset_sample is not None
            and candidate.start_sample - self._last_onset_sample > window_samples
        ):
            return (
                EventTermination.ONSET_CLUSTER,
                {
                    "since_last_onset_s": round(
                        (candidate.start_sample - self._last_onset_sample)
                        / self.sample_rate, 3
                    ),
                    "onsets_in_event": len(self._onset_samples),
                    "cluster_window_s": config.onset_cluster_window,
                },
            )

        # --- 2. change in acoustic character ----------------------------
        # Centroid and flatness are already computed per frame, so this costs
        # two running means and no extra transform.
        if features is not None:
            # Two windows, not one.  The older part of the trail represents the
            # state *before* any change; the newer part is the state now.  A
            # single trailing window straddles a transition, so its own spread
            # grows with the change it is being measured against and the
            # boundary becomes unreachable - which is why the first version
            # never fired on a genuine step.
            trail = list(self._trail)
            # The window must be FULL before a change can be claimed.
            #
            # A partially filled window compares a handful of frames against
            # whatever preceded them, and the first frames of an event are
            # exactly where the spectrum legitimately differs from the
            # background - that is the onset, not a change of acoustic
            # character.  Allowing it produced 80 ms fragment events at the
            # start of real sounds.  Detecting a change requires history.
            capacity = self._trail.maxlen or 0
            if capacity and len(trail) < capacity:
                return None
            if len(trail) < 9:
                return None
            third = max(3, len(trail) // 3)
            if len(trail) - third < 2:
                return None
            past = trail[:-third]
            recent = trail[-third:]
            for index, name in ((0, "centroid_hz"), (1, "flatness")):
                past_values = [row[index] for row in past]
                recent_values = [row[index] for row in recent]
                past_mean = sum(past_values) / len(past_values)
                recent_mean = sum(recent_values) / len(recent_values)
                variance = sum(
                    (v - past_mean) ** 2 for v in past_values
                ) / max(len(past_values) - 1, 1)
                spread = math.sqrt(variance)
                deviation = abs(recent_mean - past_mean)

                # Both thresholds are unit-free: a multiple of the observed
                # spread, and a fraction of the prior level.  An absolute floor
                # would be meaningless across features - 2.0 is a plausible
                # centroid deviation in Hz and unreachable for a flatness
                # bounded by 1.
                threshold = max(
                    config.change_point_sensitivity * spread,
                    config.change_point_relative * abs(past_mean),
                    1e-12,
                )
                if deviation > threshold:
                    return (
                        EventTermination.ACOUSTIC_CHANGE,
                        {
                            "feature": name,
                            "recent_mean": round(recent_mean, 4),
                            "prior_mean": round(past_mean, 4),
                            "prior_spread": round(spread, 4),
                            "deviation": round(deviation, 4),
                            "threshold": round(threshold, 4),
                            "at_seconds": round(
                                candidate.start_sample / self.sample_rate, 3
                            ),
                        },
                    )
        return None

    # ------------------------------------------------------------------
    def _begin(self, candidate: Candidate) -> None:
        self._state = EventState.ACTIVE
        self._frame_in_event = 0
        self._release_count = 0
        self._boundary_signalled = False
        self._trail.clear()
        self._onset_samples.clear()
        self._last_onset_sample = None
        self._note("start", activation=round(candidate.activation, 3),
                   snr_db=round(candidate.snr_db, 2))
        self._profile = EventProfile(
            onset_sample=candidate.start_sample,
            first_candidate_sample=candidate.start_sample,
            last_active_sample=candidate.start_sample,
            frames_per_second=self.frame_rate,
        )

    def _accumulate(
        self,
        candidate: Candidate,
        features: Optional[FrameFeatures],
        floor: Optional[NoiseFloorState],
    ) -> None:
        profile = self._profile
        if profile is None:
            return
        self._frame_in_event += 1
        profile.frames += 1
        profile.candidate_frames += 1 if candidate.is_candidate else 0
        profile.active_frames += 1
        profile.snr_moments.add(candidate.snr_db)
        profile.peak_snr_db = max(profile.peak_snr_db, candidate.snr_db)
        profile.mean_snr_db += candidate.snr_db
        above = candidate.snr_db >= self.config.minimum_event_snr_db
        profile.max_crest = max(profile.max_crest, _crest(candidate, features))
        profile.mean_crest += _crest(candidate, features)
        profile.mean_modulation += candidate.modulation
        profile.max_modulation = max(profile.max_modulation, candidate.modulation)
        profile.max_clipping_ratio = max(
            profile.max_clipping_ratio, candidate.clipping_ratio
        )
        profile.max_peakiness = max(
            profile.max_peakiness, candidate.spectral_peakiness
        )
        profile.mean_peakiness += candidate.spectral_peakiness
        # spectral_flux is None on the first frame of a stream, so it must be
        # guarded rather than compared.
        flux = features.spectral_flux if features is not None else None
        if flux is not None:
            profile.max_flux = max(profile.max_flux, flux)
        if candidate.spectral_peakiness >= profile.max_peakiness:
            profile.dominant_frequency_hz = candidate.dominant_frequency_hz

        for name, value in candidate.values.items():
            profile.indicators[name] = profile.indicators.get(name, 0) + 1
            profile.indicator_activation[name] = max(
                profile.indicator_activation.get(name, 0.0), value
            )

        # A new onset starts a possible footstep sequence.  The marker is a
        # flux jump, not a high crest factor: a 40 ms frame holds only a few
        # cycles of a decaying footstep thump, so a footstep measures *below*
        # the surrounding noise and never trips the crest indicator.  Flux does
        # jump at the onset, and that is the honest evidence for a sequence of
        # steps.
        onset = candidate.flux_ratio is not None and (
            candidate.flux_ratio >= 1.8
        )
        if onset:
            self._onset_samples.append(candidate.start_sample)
            self._last_onset_sample = candidate.start_sample
            gap_seconds = (
                (candidate.start_sample - profile.last_onset_sample)
                / self.sample_rate
                if profile.impulsive_onsets > 0
                else 0.0
            )
            # One onset per 80 ms minimum: a single thump can span several
            # frames and must not be counted as several steps.
            if profile.impulsive_onsets == 0 or gap_seconds > 0.08:
                if profile.impulsive_onsets > 0:
                    profile.onset_intervals.append(gap_seconds)
                profile.impulsive_onsets += 1
                profile.last_onset_sample = candidate.start_sample

        # The segmentation window needs features but *not* the noise floor.
        # Keeping it below this point made change detection silently inert
        # whenever the floor was unavailable, which is not an acceptable
        # dependency for boundary logic.
        if features is not None:
            self._trail.append(
                (features.spectral_centroid_hz, features.spectral_flatness)
            )
            # Cheap online statistics over values already in hand.  No new
            # transform: these feed the event fingerprint only.
            profile.centroid_moments.add(features.spectral_centroid_hz)
            profile.flatness_moments.add(features.spectral_flatness)
            profile.level_moments.add(features.rms_db)
            profile.level_series.append(features.rms_db)

        if features is None or floor is None:
            return

        if above:
            # Only frames that were genuinely above the floor contribute to the
            # "what is this sound when it is present" measurements.
            profile.frames_above_floor += 1
            profile.rms_above_floor_db.append(float(features.rms_db))
            profile.flatness_above_sum += features.spectral_flatness
            profile.flatness_above_count += 1
            profile.centroid_above_sum += features.spectral_centroid_hz
            profile.peakiness_above_max = max(
                profile.peakiness_above_max, features.spectral_peakiness
            )
        profile.frames_with_features += 1
        profile.peak_dbfs = max(profile.peak_dbfs, features.peak_db)
        profile.peak_level_dbfs = max(profile.peak_level_dbfs, features.peak_db)
        profile.mean_rms_dbfs += features.rms_db
        profile.mean_flatness += features.spectral_flatness
        profile.mean_centroid_hz += features.spectral_centroid_hz
        profile.mean_bandwidth_hz += features.spectral_bandwidth_hz
        profile.max_bandwidth_hz = max(
            profile.max_bandwidth_hz, features.spectral_bandwidth_hz
        )
        if candidate.snr_db >= profile.peak_snr_db:
            profile.centroid_at_peak_hz = features.spectral_centroid_hz

        low = _band(floor, LOW_BAND)
        high = _band(floor, HIGH_BAND)
        mid = _band(floor, MID_BAND)
        if low is not None:
            profile.mean_low_band_snr_db += low.snr_db
            profile.max_low_band_snr_db = max(
                profile.max_low_band_snr_db, low.snr_db
            )
        if high is not None:
            profile.mean_high_band_snr_db += high.snr_db
            profile.max_high_band_snr_db = max(
                profile.max_high_band_snr_db, high.snr_db
            )
        if mid is not None:
            profile.mean_mid_band_snr_db += mid.snr_db

    # ------------------------------------------------------------------
    def _finalise_profile(self) -> None:
        """Convert running sums into means."""
        profile = self._profile
        if profile is None or profile.frames == 0:
            return
        n = profile.frames
        profile.mean_snr_db /= n
        # These accumulators are only updated when analysis features were
        # supplied, so they are averaged by the count of frames that actually
        # carried features - not by `frames`, which is the same thing here but
        # is the wrong divisor if features are ever withheld.
        measured = profile.frames_with_features
        if measured > 0:
            profile.mean_crest /= measured
            profile.mean_modulation /= measured
            profile.mean_peakiness /= measured
            profile.mean_flatness /= measured
            profile.mean_centroid_hz /= measured
            profile.mean_bandwidth_hz /= measured
            profile.mean_low_band_snr_db /= measured
            profile.mean_high_band_snr_db /= measured
            profile.mean_mid_band_snr_db /= measured
            profile.mean_rms_dbfs /= measured
        profile.high_to_low_ratio_db = (
            profile.mean_high_band_snr_db - profile.mean_low_band_snr_db
        )
        profile.release_frames = self._release_count

    def _finish(
        self,
        reason: EventTermination = EventTermination.QUIET,
        detail: Optional[dict] = None,
    ) -> Optional[TrackedEvent]:
        self._finalise_profile()
        profile = self._profile
        assert profile is not None

        self._state = EventState.IDLE
        self._profile = None
        self._frame_in_event = 0
        self._release_count = 0

        if profile.frames < self.min_duration_frames:
            self.events_discarded_short += 1
            self._note("discard", reason="below_min_duration",
                       frames=profile.frames)
            return None

        # Stored span: pre-roll before the onset, post-roll after the last
        # active frame.  The onset itself is the detection, so pre-roll is
        # clipped at zero rather than reaching before the stream began.
        start = max(0, profile.onset_sample - self.pre_roll_frames)
        end = profile.last_active_sample + self.hop_samples + self.post_roll_frames

        event = TrackedEvent(
            start_sample=start,
            end_sample=end,
            sample_rate=self.sample_rate,
            profile=profile,
            preroll_requested_frames=min(self.pre_roll_frames, profile.onset_sample),
            preroll_missing_frames=0,  # filled in by the worker after extraction
            truncated=profile.truncated,
            termination=reason,
            boundary_detail=dict(detail or {}),
        )

        # Merging needs to know whether the *next* detection belongs to this
        # event, so a completed event is held for one frame before being
        # emitted.  That is the only place latency is added.
        if self._pending is not None:
            previous = self._pending
            gap = (
                profile.onset_sample - previous.profile.last_active_sample
            ) / self.sample_rate
            # Merging is bounded, and it must never undo a boundary that was
            # signalled deliberately.  Previously a split caused by
            # max_duration was re-fused here immediately, because the gap was
            # ~0 s, so a continuously active signal produced one unbounded
            # event and the duration cap had no effect at all.
            # The prohibition has to live on the *event*, not on the tracker:
            # a tracker-wide flag is cleared when the next event begins, so the
            # boundary that produced `previous` was forgotten before the merge
            # decision was made and the pieces were fused again.
            mergeable = (
                previous.termination not in _BOUNDARY_TERMINATIONS
                and gap <= self.config.max_merge_interval
                and not previous.truncated
            )
            if mergeable:
                self._pending = self._merge(previous, event)
                self.events_merged += 1
                self._note("merge", gap_s=round(gap, 3),
                           previous_termination=previous.termination.value)
                return None
            if gap <= self.config.merge_gap and not mergeable:
                self._note("merge_refused", gap_s=round(gap, 3),
                           reason=reason.value)
            self._pending = event
            self.events_emitted += 1
            return previous

        self._pending = event
        return None

    def _merge(self, first: TrackedEvent, second: TrackedEvent) -> TrackedEvent:
        """Fold two adjacent detections into one event."""
        a, b = first.profile, second.profile
        total = a.frames + b.frames
        if total == 0:
            total = 1

        def weighted(x: float, y: float, na: int, nb: int) -> float:
            return (x * na + y * nb) / total

        merged = EventProfile(
            onset_sample=min(a.onset_sample, b.onset_sample),
            last_active_sample=max(a.last_active_sample, b.last_active_sample),
            first_candidate_sample=min(
                a.first_candidate_sample, b.first_candidate_sample
            ),
            peak_snr_db=max(a.peak_snr_db, b.peak_snr_db),
            mean_snr_db=weighted(a.mean_snr_db, b.mean_snr_db, a.frames, b.frames),
            peak_dbfs=max(a.peak_dbfs, b.peak_dbfs),
            peak_level_dbfs=max(a.peak_level_dbfs, b.peak_level_dbfs),
            mean_rms_dbfs=min(a.mean_rms_dbfs, b.mean_rms_dbfs),
            max_crest=max(a.max_crest, b.max_crest),
            mean_crest=weighted(a.mean_crest, b.mean_crest, a.frames, b.frames),
            mean_modulation=max(a.mean_modulation, b.mean_modulation),
            max_modulation=max(a.max_modulation, b.max_modulation),
            max_clipping_ratio=max(a.max_clipping_ratio, b.max_clipping_ratio),
            max_peakiness=max(a.max_peakiness, b.max_peakiness),
            mean_peakiness=weighted(
                a.mean_peakiness, b.mean_peakiness, a.frames, b.frames
            ),
            mean_flatness=weighted(
                a.mean_flatness, b.mean_flatness, a.frames, b.frames
            ),
            mean_centroid_hz=weighted(
                a.mean_centroid_hz, b.mean_centroid_hz, a.frames, b.frames
            ),
            mean_bandwidth_hz=weighted(
                a.mean_bandwidth_hz, b.mean_bandwidth_hz, a.frames, b.frames
            ),
            max_bandwidth_hz=max(a.max_bandwidth_hz, b.max_bandwidth_hz),
            max_flux=max(a.max_flux, b.max_flux),
            mean_low_band_snr_db=weighted(
                a.mean_low_band_snr_db, b.mean_low_band_snr_db, a.frames, b.frames
            ),
            max_low_band_snr_db=max(a.max_low_band_snr_db, b.max_low_band_snr_db),
            mean_high_band_snr_db=weighted(
                a.mean_high_band_snr_db, b.mean_high_band_snr_db, a.frames, b.frames
            ),
            max_high_band_snr_db=max(a.max_high_band_snr_db, b.max_high_band_snr_db),
            mean_mid_band_snr_db=weighted(
                a.mean_mid_band_snr_db, b.mean_mid_band_snr_db,
                a.frames, b.frames,
            ),
            dominant_frequency_hz=(
                a.dominant_frequency_hz
                if a.max_peakiness >= b.max_peakiness
                else b.dominant_frequency_hz
            ),
            impulsive_onsets=a.impulsive_onsets + b.impulsive_onsets,
            onset_intervals=a.onset_intervals + b.onset_intervals,
            frames=total,
            frames_with_features=a.frames_with_features + b.frames_with_features,
            level_moments=a.level_moments.merge(b.level_moments),
            snr_moments=a.snr_moments.merge(b.snr_moments),
            centroid_moments=a.centroid_moments.merge(b.centroid_moments),
            flatness_moments=a.flatness_moments.merge(b.flatness_moments),
            level_series=deque(
                list(a.level_series) + list(b.level_series),
                maxlen=MAX_LEVEL_SERIES,
            ),
            active_frames=a.active_frames + b.active_frames,
            frames_above_floor=a.frames_above_floor + b.frames_above_floor,
            rms_above_floor_db=deque(
                list(a.rms_above_floor_db) + list(b.rms_above_floor_db),
                maxlen=MAX_RMS_SAMPLES,
            ),
            frames_per_second=(
                a.frames_per_second or b.frames_per_second
            ),
            flatness_above_sum=a.flatness_above_sum + b.flatness_above_sum,
            flatness_above_count=a.flatness_above_count + b.flatness_above_count,
            centroid_above_sum=a.centroid_above_sum + b.centroid_above_sum,
            peakiness_above_max=max(
                a.peakiness_above_max, b.peakiness_above_max
            ),
            candidate_frames=a.candidate_frames + b.candidate_frames,
            truncated=a.truncated or b.truncated,
            gapped=a.gapped or b.gapped,
        )
        for name, count in list(a.indicators.items()) + list(b.indicators.items()):
            merged.indicators[name] = merged.indicators.get(name, 0) + count
        for name, value in list(a.indicator_activation.items()) + list(
            b.indicator_activation.items()
        ):
            merged.indicator_activation[name] = max(
                merged.indicator_activation.get(name, 0.0), value
            )
        merged.high_to_low_ratio_db = (
            merged.mean_high_band_snr_db - merged.mean_low_band_snr_db
        )

        return TrackedEvent(
            start_sample=min(first.start_sample, second.start_sample),
            end_sample=max(first.end_sample, second.end_sample),
            sample_rate=self.sample_rate,
            profile=merged,
            preroll_requested_frames=max(
                first.preroll_requested_frames, second.preroll_requested_frames
            ),
            truncated=merged.truncated,
            merged_count=first.merged_count + second.merged_count + 1,
            # A merged event ends when the *later* piece ended; the earlier
            # reason no longer describes it.
            termination=second.termination,
            boundary_detail=dict(second.boundary_detail),
        )

    # ------------------------------------------------------------------
    def flush(self) -> list[TrackedEvent]:
        """Finish the stream and return every event still owed.

        For end of stream.  An event still in progress is finalised with the
        release period treated as satisfied.

        Returns a list rather than a single event because the hold-for-merge
        step means two events can be outstanding at once: finalising the last
        one releases the one before it.  Returning only the newest would
        silently drop a real event, which is exactly the failure that a single
        return value invites.
        """
        out: list[TrackedEvent] = []
        if self._state is not EventState.IDLE and self._profile is not None:
            self._note("terminate", reason=EventTermination.END_OF_STREAM.value)
            completed = self._finish(EventTermination.END_OF_STREAM)
            if completed is not None:
                out.append(completed)
        pending = self._pending
        self._pending = None
        if pending is not None:
            self.events_emitted += 1
            out.append(pending)
        return out


def _crest(candidate: Candidate, features: Optional[FrameFeatures]) -> float:
    return features.crest_factor if features is not None else 0.0


def _band(floor: NoiseFloorState, label: str):
    try:
        return floor.band(label)
    except KeyError:
        return None
