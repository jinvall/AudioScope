"""Central configuration for Audio Microscope.

Architecture note (docs/ARCHITECTURE.md section 10): constants must not be
scattered through the codebase.  Every tunable lives here.

The configuration is a plain dataclass so it can be constructed in tests with
no file system access, serialised to JSON, and validated in one place.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any

# --------------------------------------------------------------------------
# Internal audio format (docs/AUDIO_PIPELINE.md section 1)
# --------------------------------------------------------------------------
INTERNAL_SAMPLE_RATE = 48_000
INTERNAL_CHANNELS = 1
INTERNAL_DTYPE = "float32"

# Docs allow 10-120 seconds for the rolling buffer (AGENTS.md section 5).
RING_BUFFER_MIN_SECONDS = 10.0
RING_BUFFER_MAX_SECONDS = 120.0


class ConfigError(ValueError):
    """Raised when configuration values are inconsistent or out of range."""


def _validate_range(name: str, value: float, low: float, high: float) -> None:
    if not (low <= value <= high):
        raise ConfigError(
            f"{name} must be between {low} and {high}, got {value}"
        )


@dataclass
class AudioConfig:
    """Capture-side settings."""

    # None -> ask PortAudio for the system default input.
    device: str | int | None = None
    # We ask for this rate.  If the device refuses, we open at the device's
    # native rate and resample once at the input boundary.
    sample_rate: int = INTERNAL_SAMPLE_RATE
    channels: int = INTERNAL_CHANNELS
    # Frames per callback.  Small enough to keep latency low, large enough to
    # amortise the callback overhead.
    block_size: int = 1024
    # Bounded hand-off queue to the recorder/analysis workers.  Bounded so a
    # stalled disk cannot grow memory without limit (docs/PERFORMANCE.md 7).
    queue_max_blocks: int = 256


@dataclass
class BufferConfig:
    """Rolling buffer settings (docs/AUDIO_PIPELINE.md section 3)."""

    seconds: float = 30.0

    def __post_init__(self) -> None:
        _validate_range(
            "buffer.seconds",
            self.seconds,
            RING_BUFFER_MIN_SECONDS,
            RING_BUFFER_MAX_SECONDS,
        )


@dataclass
class EventConfig:
    """Event window settings (AGENTS.md section 5).

    Phase 1 does not detect events, but the capture layer must already retain
    enough history to serve the configured pre-roll, so these live here now.
    """

    pre_roll_seconds: float = 5.0
    post_roll_seconds: float = 5.0
    max_event_seconds: float = 30.0

    def __post_init__(self) -> None:
        _validate_range("event.pre_roll_seconds", self.pre_roll_seconds, 0.0, 60.0)
        _validate_range("event.post_roll_seconds", self.post_roll_seconds, 0.0, 60.0)
        _validate_range("event.max_event_seconds", self.max_event_seconds, 0.1, 600.0)
        if self.pre_roll_seconds >= self.buffer_required_seconds:
            raise ConfigError(
                "event.pre_roll_seconds "
                f"({self.pre_roll_seconds}s) must be smaller than the ring "
                "buffer duration, otherwise history cannot be guaranteed"
            )

    @property
    def buffer_required_seconds(self) -> float:
        return self.pre_roll_seconds + 5.0


@dataclass
class RecordConfig:
    """Continuous raw recording (docs/DATA_AND_STORAGE.md section 6)."""

    enabled: bool = True
    directory: str = "recordings"
    # 15 minutes per file (AGENTS.md section 22).
    chunk_seconds: float = 900.0
    # Never delete evidence without an explicit policy.
    retention_days: int | None = None

    def __post_init__(self) -> None:
        _validate_range("record.chunk_seconds", self.chunk_seconds, 1.0, 86_400.0)
        if self.retention_days is not None and self.retention_days <= 0:
            raise ConfigError("record.retention_days must be positive or None")


@dataclass
class PlaybackConfig:
    device: str | int | None = None
    sample_rate: int = INTERNAL_SAMPLE_RATE
    volume: float = 1.0
    loop: bool = False

    def __post_init__(self) -> None:
        _validate_range("playback.volume", self.volume, 0.0, 1.0)


@dataclass
class AnalysisConfig:
    """Framing, features, and noise-floor behaviour (Phase 2).

    docs/AUDIO_PIPELINE.md section 4 requires a configurable frame of roughly
    20-50 ms with enough overlap for a stable spectrum.  The default of 40 ms
    with a 10 ms hop gives 1920-point frames at 48 kHz, a 25 Hz bin width, and
    100 analyses per second.

    The floor is asymmetric by design (AGENTS.md section 8): it may fall
    quickly when the room gets quieter, but it may only rise slowly, and it
    refuses to learn from frames that look like events.
    """

    frame_ms: float = 40.0
    hop_ms: float = 10.0
    # Hann is the default: low side-lobe leakage, which matters because band
    # energies and a noise floor are sensitive to leakage from loud neighbours.
    window: str = "hann"
    # Fraction of spectral energy below which the rolloff frequency sits.
    rolloff_percentile: float = 0.85

    # Noise floor, in dB relative to full scale.
    # Rise is deliberately slow: an event must not become the new floor.
    floor_rise_db_per_sec: float = 1.5
    floor_fall_db_per_sec: float = 12.0
    # Minimum-statistics parameters, in seconds.
    min_stats_window_sec: float = 3.0
    # The sub-window is how long a quiet stretch must be before the estimator
    # can see it: a 0.5 s sub-window cannot fit inside a 0.25 s pause between
    # whispered phrases, so the floor never finds the room beneath a sustained
    # sound and its signal-to-noise ratio collapses to zero.
    #
    # Measured on this system (quantile 0.2), against a whisper occupying 70%
    # of a 12 s window with 0.25 s pauses:
    #
    #   sub-window   whisper presence   footsteps   empty room   phantom band SNR
    #      0.50 s          0.11             0.04        0.00         0.02 dB
    #      0.25 s          0.27             0.04        0.00         0.06 dB
    #      0.15 s          0.53             0.04        0.00         0.14 dB
    #      0.10 s          0.65             0.04        0.00         0.19 dB
    #      0.05 s          0.67             0.04        0.00         0.30 dB
    #
    # 0.10 s is the chosen point: it recovers a sustained sound while leaving
    # the intermittent and empty cases untouched.  Going shorter buys almost
    # nothing more and doubles the residual bias.
    min_stats_subwindow_sec: float = 0.1
    # A frame this far above the floor is treated as an event and is not
    # allowed to raise the floor.
    transient_gate_db: float = 9.0
    # Which quantile of the accepted sub-window levels becomes the raw
    # estimate.  0 is the absolute minimum (biased low, badly so for narrow
    # bands) and 1 is the mean (absorbs any sound present most of the time).
    # 0.2 sits between: it still tracks the quiet part of the window so a
    # sustained sound stays visible, without the sample-size bias of a minimum.
    floor_quantile: float = 0.2
    # ...but a genuinely louder room must eventually become the new floor.
    # If the level stays above the gate for this long, it is reclassified as
    # the environment rather than an event, and the floor is allowed to follow.
    # Without this the gate would freeze the floor forever in a noisier room.
    floor_sustain_sec: float = 3.0

    def __post_init__(self) -> None:
        _validate_range("analysis.frame_ms", self.frame_ms, 10.0, 500.0)
        _validate_range("analysis.hop_ms", self.hop_ms, 1.0, 500.0)
        if self.hop_ms > self.frame_ms:
            raise ConfigError(
                f"analysis.hop_ms ({self.hop_ms}ms) must not exceed "
                f"analysis.frame_ms ({self.frame_ms}ms); a hop longer than the "
                "frame would skip audio"
            )
        if self.window not in ANALYSIS_WINDOWS:
            raise ConfigError(
                f"analysis.window must be one of "
                f"{sorted(ANALYSIS_WINDOWS)}, got {self.window!r}"
            )
        _validate_range(
            "analysis.rolloff_percentile", self.rolloff_percentile, 0.0, 1.0
        )
        if self.rolloff_percentile <= 0.0:
            raise ConfigError(
                "analysis.rolloff_percentile must be greater than 0"
            )
        if self.floor_rise_db_per_sec <= 0:
            raise ConfigError("analysis.floor_rise_db_per_sec must be positive")
        if self.floor_fall_db_per_sec <= 0:
            raise ConfigError("analysis.floor_fall_db_per_sec must be positive")
        if self.floor_fall_db_per_sec < self.floor_rise_db_per_sec:
            raise ConfigError(
                "analysis.floor_fall_db_per_sec must be at least "
                "analysis.floor_rise_db_per_sec; the floor must be able to "
                "follow a quieting room at least as fast as it rises"
            )
        _validate_range(
            "analysis.min_stats_window_sec", self.min_stats_window_sec, 0.1, 60.0
        )
        _validate_range("analysis.floor_quantile", self.floor_quantile, 0.0, 1.0)
        _validate_range(
            "analysis.min_stats_subwindow_sec", self.min_stats_subwindow_sec,
            0.02, 10.0,
        )
        if self.min_stats_subwindow_sec > self.min_stats_window_sec:
            raise ConfigError(
                "analysis.min_stats_subwindow_sec must not exceed "
                "analysis.min_stats_window_sec"
            )
        _validate_range(
            "analysis.transient_gate_db", self.transient_gate_db, 0.0, 60.0
        )
        _validate_range(
            "analysis.floor_sustain_sec", self.floor_sustain_sec, 0.0, 300.0
        )

    # ------------------------------------------------------------------
    def frame_samples_at(self, sample_rate: int) -> int:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        return max(8, int(round(self.frame_ms * sample_rate / 1000.0)))

    def hop_samples_at(self, sample_rate: int) -> int:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        return max(1, int(round(self.hop_ms * sample_rate / 1000.0)))

    @property
    def overlap_ratio(self) -> float:
        return 1.0 - (self.hop_ms / self.frame_ms)


# Window functions available to the analysis stage.
ANALYSIS_WINDOWS = ("hann", "hamming", "blackman", "rect", "none")


_BYTE_UNITS = {
    "": 1,
    "b": 1,
    "k": 1024, "kb": 1024, "kib": 1024,
    "m": 1024 ** 2, "mb": 1024 ** 2, "mib": 1024 ** 2,
    "g": 1024 ** 3, "gb": 1024 ** 3, "gib": 1024 ** 3,
    "t": 1024 ** 4, "tb": 1024 ** 4, "tib": 1024 ** 4,
}


def parse_byte_size(value: Any) -> int:
    """Parse a byte size from a number or a human-readable string.

    Accepts ``524288000``, ``"500MB"``, ``"5GB"``, ``"1.5 GiB"``, ``"2g"``.

    Note the units: ``MB`` here means 1024**2, which is the convention the
    project already uses when it says 500 MB and writes 524288000.  ``MiB``
    is accepted as an explicit synonym so the ambiguity is available to the
    reader rather than hidden in the parsing.  There is no decimal (10**6)
    interpretation, because a 5% discrepancy between the configured cap and
    the intended one is exactly the sort of thing that only shows up after a
    long run of unattended recording.
    """
    if isinstance(value, bool):
        raise ConfigError(f"invalid byte size: {value!r}")
    if isinstance(value, (int, float)):
        if value < 0:
            raise ConfigError(f"byte size must not be negative: {value!r}")
        return int(value)
    if not isinstance(value, str):
        raise ConfigError(f"invalid byte size: {value!r}")

    text = value.strip().replace("_", "")
    if not text:
        raise ConfigError("empty byte size")
    number = ""
    index = 0
    while index < len(text) and (text[index].isdigit() or text[index] in ".+-"):
        number += text[index]
        index += 1
    unit = text[index:].strip().lower()
    if not number:
        raise ConfigError(f"invalid byte size: {value!r}")
    if unit not in _BYTE_UNITS:
        raise ConfigError(
            f"unknown size unit {unit!r} in {value!r}; use one of "
            f"{sorted(u for u in _BYTE_UNITS if u)}"
        )
    try:
        magnitude = float(number)
    except ValueError as exc:
        raise ConfigError(f"invalid byte size: {value!r}") from exc
    if magnitude < 0:
        raise ConfigError(f"byte size must not be negative: {value!r}")
    return int(round(magnitude * _BYTE_UNITS[unit]))


def format_byte_size(num_bytes: float) -> str:
    """Render bytes for display, e.g. ``472.0 MB``.  Informational only."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TB"


@dataclass
class AudioRetentionConfig:
    """Byte-capped retention for stored event audio.

    The governing idea, and the reason this is separate from
    :class:`RecordConfig`: **the fingerprint database is the application's
    long-term memory and audio is bounded evidence.**  Every retained event
    keeps its fingerprint, its measurements and its review history
    indefinitely.  Only the audio file is disposable, and only when bytes
    demand it.

    Limits are in **bytes, not event counts**, because event duration and
    encoded size vary by orders of magnitude: 100 five-second events and 100
    thirty-second events are the same "100 events" and very different
    amounts of storage.  A count-based cap would be a cap on the wrong thing.

    Two independent limits apply to stored audio only - never to
    fingerprints, feature vectors, metadata, indexes or classification
    records:

    * ``per_class_cap_bytes`` - each classification gets its own budget, so
      one dominant class cannot consume the whole store and starve the rest.
    * ``total_cap_bytes`` - the global bound.
    """

    enabled: bool = True

    # 500 MiB per classification.
    per_class_cap_bytes: int = 500 * 1024 ** 2
    # 5 GiB in total.
    total_cap_bytes: int = 5 * 1024 ** 3

    # Evict redundancy rather than age.  The default keeps a representative
    # spread of each class instead of the most recent N, which is the whole
    # point of a bounded evidence cache.  False falls back to oldest-first.
    preserve_representative_audio: bool = True

    # How often the accounting is re-derived from the filesystem.  The hot
    # path never scans the tree; this is the safety net that corrects drift
    # from an external deletion, a crash, or a restore from backup.
    reconcile_interval_seconds: float = 900.0

    # Events a human has judged are never evicted automatically.  Both
    # directions are protected: a confirmed example and a rejected negative
    # example are each irreplaceable as a labelled data point, and the
    # project treats a rejection as evidence rather than as rubbish.
    protect_human_decisions: bool = True

    # Bytes held back from the caps when deciding whether there is room.  A
    # small margin means a burst of near-simultaneous events does not land
    # exactly on the cap and immediately trip eviction on the next event.
    headroom_bytes: int = 0

    # Never reduce a classification to zero audio files.
    #
    # Without this, a single recording larger than its class cap - a long
    # event, or a class that happens to have one recording - gets deleted by
    # the next reconciliation with nothing to replace it, destroying the
    # newest evidence in the class in exchange for satisfying a storage
    # policy.  The overage is reported instead.  Every *write* still enforces
    # the caps strictly, because there the incoming file has to fit, and
    # making room for it is the whole point of reserving first.
    min_files_per_class: int = 1

    def __post_init__(self) -> None:
        self.per_class_cap_bytes = parse_byte_size(self.per_class_cap_bytes)
        self.total_cap_bytes = parse_byte_size(self.total_cap_bytes)
        self.headroom_bytes = parse_byte_size(self.headroom_bytes)

        if self.per_class_cap_bytes <= 0:
            raise ConfigError("audio_retention.per_class_cap_bytes must be positive")
        if self.total_cap_bytes <= 0:
            raise ConfigError("audio_retention.total_cap_bytes must be positive")
        if self.per_class_cap_bytes > self.total_cap_bytes:
            raise ConfigError(
                "audio_retention.per_class_cap_bytes "
                f"({self.per_class_cap_bytes}) must not exceed "
                f"audio_retention.total_cap_bytes ({self.total_cap_bytes}); a "
                "per-class cap above the global cap cannot be enforced"
            )
        _validate_range(
            "audio_retention.reconcile_interval_seconds",
            self.reconcile_interval_seconds, 1.0, 86_400.0,
        )
        if self.min_files_per_class < 0:
            raise ConfigError(
                "audio_retention.min_files_per_class must not be negative"
            )
        if self.headroom_bytes >= self.total_cap_bytes:
            raise ConfigError(
                "audio_retention.headroom_bytes must be smaller than "
                "audio_retention.total_cap_bytes"
            )

    def to_dict(self) -> dict[str, Any]:
        # Byte values are always explicit integers, even if the configuration
        # file used "500MB"; the readable form is an input convenience.
        return {
            "enabled": self.enabled,
            "per_class_cap_bytes": self.per_class_cap_bytes,
            "total_cap_bytes": self.total_cap_bytes,
            "preserve_representative_audio": self.preserve_representative_audio,
            "reconcile_interval_seconds": self.reconcile_interval_seconds,
            "protect_human_decisions": self.protect_human_decisions,
            "headroom_bytes": self.headroom_bytes,
            "min_files_per_class": self.min_files_per_class,
        }



@dataclass
class DetectionConfig:
    """Candidate detection and event tracking (Phase 3).

    docs/AUDIO_PIPELINE.md section 8 requires hysteresis rather than a bare
    threshold, otherwise one real event produces a candidate on every other
    frame.  The onset and continuation thresholds differ so the state machine
    does not chatter at the boundary.

    docs/DETECTION_AND_CLASSIFICATION.md section 2 asks for deliberately broad
    detection: it is better to record an unknown event than to miss a quiet one
    because the detector was too strict.  The thresholds here are therefore
    permissive, and the classifier - not the detector - decides what an event
    is.
    """

    # Hysteresis. Onset must be clearly above the continuation level.
    onset_threshold: float = 0.30
    continuation_threshold: float = 0.16

    # Duration limits, in seconds.
    min_duration: float = 0.05
    max_duration: float = 30.0
    # How long the signal must stay below the continuation threshold before the
    # event is considered finished (docs/AUDIO_PIPELINE.md section 10).
    release_timeout: float = 0.40
    # Two events closer together than this are one event.  Bounded separately
    # by max_merge_interval below, which is the value that actually decides.
    merge_gap: float = 0.30

    # ------------------------------------------------------------------
    # Event segmentation
    #
    # Termination used to depend on the activation dropping below the
    # continuation threshold for release_timeout.  On a signal that never
    # falls below the floor - which any continuously active environment
    # produces, since the floor is adaptive - that condition never occurs, and
    # every onset resets the timer again, so the event could not end.  A second
    # problem compounded it: max_duration split the event, and merge_gap
    # immediately re-fused the pieces, so the cap had no net effect.
    #
    # Segmentation therefore gets two further boundary conditions, both
    # derived from features the analysis stage already computes, and neither
    # requiring silence.  All values are explicit and generic.
    # ------------------------------------------------------------------

    # Trailing window used to characterise the recent acoustic state, in
    # seconds.  A change is measured against this, not against the whole event.
    change_point_window: float = 1.5
    # How far the current value may sit from the trailing mean, as a multiple
    # of the trailing spread, before it counts as a change in acoustic
    # character.  Small enough to notice a real transition, large enough that
    # ordinary fluctuation does not trip it.
    change_point_sensitivity: float = 3.0
    # A deviation also has to clear a multiple of the trailing mean.  Both
    # terms are unit-free on purpose: the features used are a centroid in Hz
    # and a flatness bounded by 1, so any absolute floor is meaningless for one
    # of them and unreachable for the other.
    change_point_relative: float = 0.35
    # Onsets further apart than this belong to different temporal clusters,
    # and the event ends between them.  Guarded by min_onsets_for_clustering so
    # a signal with one or two onsets is never fragmented.
    onset_cluster_window: float = 2.0
    min_onsets_for_clustering: int = 3
    # Hard ceiling on how far apart two events may be and still be merged.
    # Kept below the change-point and clustering windows so a deliberate
    # boundary cannot be undone by merging, which is what defeated
    # max_duration before.
    max_merge_interval: float = 1.0

    # Candidate indicators, in 0..1. Each is a *relative* measure against the
    # adaptive noise floor or against its own recent history, never a fixed
    # dBFS value (AGENTS.md section 8).
    snr_full_db: float = 18.0        # SNR treated as "fully present"
    low_band_full_db: float = 15.0   # 80-250 Hz above its own floor
    high_band_full_db: float = 12.0  # 2-8 kHz above its own floor
    # Envelope modulation depth, which is bounded in 0..1. Measured: quiet
    # noise 0.07, a footstep sequence 0.98, modulated speech 0.97. So this
    # ramp separates "quiet" from "modulated" and deliberately does *not*
    # separate footsteps from speech; presence does that.
    modulation_low: float = 0.15
    modulation_full: float = 0.45

    # Spectral flux, as a multiple of its own running median. Measured on this
    # system: background noise sits at about 0.376 (successive noise spectra are
    # uncorrelated, so half-rectified change is large) and a footstep onset
    # spikes to about 0.999, a factor of 2.7. An absolute threshold cannot
    # work here at all - it would flag the background itself - so flux is only
    # meaningful relative to its own history.
    flux_ratio_low: float = 1.3      # this multiple of the median is the onset
    flux_ratio_full: float = 2.5     # this multiple counts as fully present

    # Crest factor, as an absolute value. Measured: Gaussian noise over a
    # 1920-sample frame peaks at about 3.6 median and 4.6 maximum, so anything
    # below 5 is not evidence of a transient. A footstep thump measures *lower*
    # than the noise around it (about 2.5-3.9), because a 40 ms frame holds
    # only a few cycles of a decaying 90 Hz thump. Crest therefore detects
    # sharp, broadband transients - clicks, taps, glass - and says nothing
    # about footsteps; the classifier is not allowed to rely on it for those.
    crest_low: float = 5.0
    crest_full: float = 15.0

    # Window over which the running medians for flux and crest are kept.
    baseline_window: float = 3.0

    # Envelope window for the modulation measure, in seconds. Speech is
    # modulated at roughly 2-8 Hz, so the window must span several cycles.
    modulation_window: float = 1.0

    # Ignore candidates below this level outright, to avoid storing the
    # constant background of an empty room as a stream of events.
    minimum_event_snr_db: float = 6.0

    # ------------------------------------------------------------------
    # Whisper discrimination
    #
    # A whisper is quiet *and sustained*.  A sequence of footsteps is quiet on
    # average too, because the gaps pull the mean down, so a mean level cannot
    # separate them; presence can.  These are the presence range over which a
    # whisper goes from "not present enough" to "clearly continuous".
    # ------------------------------------------------------------------
    # Calibrated against the measured separation rather than guessed. A
    # footstep sequence runs at presence 0.04, a single footstep 0.04-0.09, and
    # a sustained whispered phrase with brief pauses between syllables and
    # phrases at 0.50-0.65. So 0.15 sits well clear of the impulsive cases and
    # 0.45 is reached by a realistic whisper. Setting these at 0.40/0.80 - which
    # is where they started - meant no achievable whisper ever cleared the
    # "strong presence" point, so every sustained whisper was scored as if it
    # were intermittent.
    whisper_presence_low: float = 0.15
    whisper_presence_full: float = 0.45
    # How much of the whisper score survives at zero presence.  Not zero: a
    # single loud frame in an otherwise quiet event is not a whisper either, but
    # the score should be suppressed rather than annihilated so the evidence
    # stays inspectable.
    whisper_presence_floor_weight: float = 0.15
    # Impulse density at which the whisper penalty reaches full strength, in
    # onsets *per analysis frame* (so the value does not depend on the hop
    # size).  A gentle, gradual penalty - never an exclusion: whispered speech
    # has consonant transients and must stay classifiable.
    #
    # Worked examples at 100 frames/s:
    #   3 footsteps over 3.5 s  -> 3/350 = 0.009  (small penalty; the
    #                                presence term is what rejects this)
    #   5 consonants over 2.0 s -> 5/200 = 0.025  (noticeable but survivable)
    #   an onset every 0.2 s   -> 1/20  = 0.050  (full penalty)
    whisper_impulse_density_low: float = 0.004
    whisper_impulse_density_full: float = 0.05
    whisper_impulse_penalty: float = 0.30

    # The movement family (movement, clothing, scraping) is characterised by
    # the *absence* of sharp impulsive onsets: cloth rustles and body movement
    # are diffuse, while a footstep or a knock is a distinct transient.  This
    # scales those rules down when the event is dominated by onsets, which is
    # the mirror image of the whisper rule and stops a footstep sequence from
    # being read as clothing.
    family_onset_density_low: float = 0.005
    family_onset_density_full: float = 0.03
    family_onset_suppression: float = 0.70

    def __post_init__(self) -> None:
        _validate_range(
            "detection.onset_threshold", self.onset_threshold, 0.0, 1.0
        )
        _validate_range(
            "detection.continuation_threshold",
            self.continuation_threshold, 0.0, 1.0,
        )
        if self.continuation_threshold >= self.onset_threshold:
            raise ConfigError(
                "detection.continuation_threshold must be lower than "
                "detection.onset_threshold; otherwise there is no hysteresis "
                "and the event state will chatter"
            )
        _validate_range("detection.min_duration", self.min_duration, 0.0, 60.0)
        _validate_range("detection.max_duration", self.max_duration, 0.1, 600.0)
        if self.min_duration >= self.max_duration:
            raise ConfigError(
                "detection.min_duration must be less than detection.max_duration"
            )
        _validate_range(
            "detection.release_timeout", self.release_timeout, 0.0, 30.0
        )
        _validate_range("detection.merge_gap", self.merge_gap, 0.0, 30.0)
        _validate_range(
            "detection.change_point_window", self.change_point_window, 0.1, 60.0
        )
        _validate_range(
            "detection.change_point_sensitivity",
            self.change_point_sensitivity, 0.1, 100.0,
        )
        _validate_range(
            "detection.change_point_relative",
            self.change_point_relative, 0.0, 100.0,
        )
        _validate_range(
            "detection.onset_cluster_window",
            self.onset_cluster_window, 0.0, 120.0,
        )
        if self.min_onsets_for_clustering < 1:
            raise ConfigError(
                "detection.min_onsets_for_clustering must be at least 1"
            )
        _validate_range(
            "detection.max_merge_interval", self.max_merge_interval, 0.0, 60.0
        )
        _validate_range("detection.snr_full_db", self.snr_full_db, 1.0, 120.0)
        _validate_range("detection.flux_ratio_low", self.flux_ratio_low, 1.0, 50.0)
        _validate_range(
            "detection.flux_ratio_full", self.flux_ratio_full, 1.0, 100.0
        )
        if self.flux_ratio_full <= self.flux_ratio_low:
            raise ConfigError(
                "detection.flux_ratio_full must exceed detection.flux_ratio_low"
            )
        _validate_range("detection.crest_low", self.crest_low, 1.0, 100.0)
        _validate_range(
            "detection.baseline_window", self.baseline_window, 0.2, 60.0
        )
        _validate_range(
            "detection.low_band_full_db", self.low_band_full_db, 1.0, 120.0
        )
        _validate_range(
            "detection.high_band_full_db", self.high_band_full_db, 1.0, 120.0
        )
        if self.crest_full <= self.crest_low:
            raise ConfigError(
                "detection.crest_full must exceed detection.crest_low"
            )
        _validate_range(
            "detection.modulation_low", self.modulation_low, 0.0, 1.0
        )
        _validate_range(
            "detection.modulation_full", self.modulation_full, 0.01, 1.0
        )
        if self.modulation_full <= self.modulation_low:
            raise ConfigError(
                "detection.modulation_full must exceed detection.modulation_low"
            )
        _validate_range(
            "detection.modulation_window", self.modulation_window, 0.1, 10.0
        )
        _validate_range(
            "detection.minimum_event_snr_db",
            self.minimum_event_snr_db, 0.0, 60.0,
        )
        _validate_range(
            "detection.whisper_presence_low",
            self.whisper_presence_low, 0.0, 1.0,
        )
        _validate_range(
            "detection.whisper_presence_full",
            self.whisper_presence_full, 0.0, 1.0,
        )
        if self.whisper_presence_full <= self.whisper_presence_low:
            raise ConfigError(
                "detection.whisper_presence_full must exceed "
                "detection.whisper_presence_low"
            )
        _validate_range(
            "detection.whisper_presence_floor_weight",
            self.whisper_presence_floor_weight, 0.0, 1.0,
        )
        _validate_range(
            "detection.whisper_impulse_density_low",
            self.whisper_impulse_density_low, 0.0, 100.0,
        )
        _validate_range(
            "detection.whisper_impulse_density_full",
            self.whisper_impulse_density_full, 0.0, 200.0,
        )
        if (
            self.whisper_impulse_density_full
            <= self.whisper_impulse_density_low
        ):
            raise ConfigError(
                "detection.whisper_impulse_density_full must exceed "
                "detection.whisper_impulse_density_low"
            )
        _validate_range(
            "detection.whisper_impulse_penalty",
            self.whisper_impulse_penalty, 0.0, 1.0,
        )
        _validate_range(
            "detection.family_onset_density_low",
            self.family_onset_density_low, 0.0, 1.0,
        )
        _validate_range(
            "detection.family_onset_density_full",
            self.family_onset_density_full, 0.0, 1.0,
        )
        if self.family_onset_density_full <= self.family_onset_density_low:
            raise ConfigError(
                "detection.family_onset_density_full must exceed "
                "detection.family_onset_density_low"
            )
        _validate_range(
            "detection.family_onset_suppression",
            self.family_onset_suppression, 0.0, 1.0,
        )


@dataclass
class SeparationConfig:
    """Query-based source separation (Phase 4).

    The separation model is AudioSep: a CLAP text encoder that turns the
    query into an embedding, and a ResUNet30 that separates the mixture
    against that embedding.  It is a real model, running its own documented
    inference path, on CPU.

    Two measured facts shape this whole section, both from a real 5 s event
    on this host (4 cores, no GPU):

    * **Cost.** 4.5 GB resident once loaded, 35 s to load, and 25 s of CPU
      to separate 5 s of audio - a realtime ratio of 5.1.
    * **Consequence.** This cannot share a process with capture.  A 4.5 GB
      model and a four-thread torch pool inside the capture process would
      compete directly with the audio callback for memory bandwidth and
      cores, and the callback is the one thing that must never be late.

    So separation runs in its **own OS process** (``torch_threads`` threads,
    ``nice_level``) reached over a line protocol by
    :class:`app.separation.model.ModelProcess`.  Process isolation is what
    actually protects the live pipeline here; lowering the analysis rate, the
    alternative considered, would have cost detection quality across the whole
    application to solve a problem isolation solves for free.

    The model stays resident between jobs because the 35 s load dominates the
    25 s of work.  ``model_idle_unload_seconds`` releases it when idle, and
    that is monitored rather than assumed (docs/SOURCE_SEPARATION.md 6).
    """

    enabled: bool = True

    # ------------------------------------------------------------------
    # Model location.  Paths are relative to the project root unless
    # absolute, so a checkout can be moved without editing configuration.
    # ------------------------------------------------------------------
    venv_python: str = "venv-sep/bin/python"
    model_dir: str = "third_party/AudioSep"
    config_yaml: str = "config/audiosep_base.yaml"
    checkpoint_path: str = "checkpoint/audiosep_base_4M_steps.ckpt"
    # The CLAP text encoder checkpoint.  AudioSep's CLAP_Encoder takes this
    # as a *relative* path resolved against the model directory, so it is
    # named here only so the preflight check can verify it exists.
    query_encoder_checkpoint: str = (
        "checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt"
    )

    # AudioSep operates at 32 kHz and resamples its input itself.  Recorded
    # in metadata so a result is never mistaken for a 48 kHz measurement.
    model_sample_rate: int = 32_000

    # ------------------------------------------------------------------
    # CPU protection (docs/SOURCE_SEPARATION.md section 8)
    # ------------------------------------------------------------------
    # One worker, deliberately.  Live monitoring continues regardless.
    workers: int = 1
    # One core is left to the capture and analysis processes.  This is the
    # deliberate budget decision: analysis (0.63 ms/frame) plus detection
    # (0.65 ms/frame) is already about 1.3 ms/frame on 100 frames/s, so the
    # separation process is capped rather than the live path being slowed.
    torch_threads: int = 3
    # Positive nice value, so under load the kernel prefers the capture and
    # analysis processes over separation.
    nice_level: int = 10
    queue_max_jobs: int = 8
    # Hard ceiling on one job's wall clock.  A job that exceeds it is
    # abandoned and reported as a failure; the model process is killed so it
    # cannot leak a stuck worker.
    job_timeout_seconds: float = 1800.0
    # 0 disables unloading.  Default 15 minutes: long enough to survive a
    # review session, short enough that an abandoned GUI does not hold
    # 4.5 GB until the machine is rebooted.
    model_idle_unload_seconds: float = 900.0

    # ------------------------------------------------------------------
    # Job bounds
    # ------------------------------------------------------------------
    # At a measured 5.1x realtime, 30 s of input is about 2.5 minutes of CPU.
    # Longer separations are refused rather than silently truncated.
    max_input_seconds: float = 30.0
    min_input_seconds: float = 0.5

    # ------------------------------------------------------------------
    # Optional enhancement (AGENTS.md section 17)
    # ------------------------------------------------------------------
    enable_enhancement: bool = True
    # Conservative: the isolated signal is evidence and is never pushed
    # towards full scale.  -3 dBFS leaves headroom and cannot clip.
    enhance_target_peak_dbfs: float = -3.0
    # Enhancement only ever writes *enhanced.wav*.  isolated.wav is always
    # the raw model output.
    enhance_remove_dc: bool = True

    # ------------------------------------------------------------------
    # Output validation thresholds (docs/SOURCE_SEPARATION.md section 9)
    # ------------------------------------------------------------------
    # A separation that removes everything is a failure, not a quiet result.
    min_output_peak_dbfs: float = -80.0
    # Fraction of samples at or beyond full scale considered catastrophic.
    max_clipped_fraction: float = 0.001

    def __post_init__(self) -> None:
        if not isinstance(self.workers, int) or self.workers < 1:
            raise ConfigError("separation.workers must be at least 1")
        if self.torch_threads < 1:
            raise ConfigError("separation.torch_threads must be at least 1")
        _validate_range("separation.nice_level", self.nice_level, 0, 19)
        if self.queue_max_jobs < 1:
            raise ConfigError("separation.queue_max_jobs must be at least 1")
        _validate_range(
            "separation.job_timeout_seconds",
            self.job_timeout_seconds, 10.0, 86_400.0,
        )
        _validate_range(
            "separation.model_idle_unload_seconds",
            self.model_idle_unload_seconds, 0.0, 86_400.0,
        )
        _validate_range(
            "separation.max_input_seconds",
            self.max_input_seconds, 1.0, 600.0,
        )
        _validate_range(
            "separation.min_input_seconds",
            self.min_input_seconds, 0.01, 60.0,
        )
        if self.min_input_seconds >= self.max_input_seconds:
            raise ConfigError(
                "separation.min_input_seconds must be less than "
                "separation.max_input_seconds"
            )
        if self.model_sample_rate < 8_000:
            raise ConfigError(
                "separation.model_sample_rate must be at least 8000"
            )
        _validate_range(
            "separation.enhance_target_peak_dbfs",
            self.enhance_target_peak_dbfs, -24.0, 0.0,
        )
        _validate_range(
            "separation.min_output_peak_dbfs",
            self.min_output_peak_dbfs, -200.0, -20.0,
        )
        _validate_range(
            "separation.max_clipped_fraction",
            self.max_clipped_fraction, 0.0, 0.5,
        )


@dataclass
class AppConfig:
    """Top-level configuration object."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    event: EventConfig = field(default_factory=EventConfig)
    record: RecordConfig = field(default_factory=RecordConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    audio_retention: AudioRetentionConfig = field(
        default_factory=AudioRetentionConfig
    )
    output_dir: str = "events"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    _SECTIONS = {
        "audio": "AudioConfig",
        "buffer": "BufferConfig",
        "event": "EventConfig",
        "record": "RecordConfig",
        "playback": "PlaybackConfig",
        "analysis": "AnalysisConfig",
        "detection": "DetectionConfig",
        "separation": "SeparationConfig",
        "audio_retention": "AudioRetentionConfig",
    }

    def __post_init__(self) -> None:
        """Coerce dict sections to their dataclasses, then validate.

        Lets callers write ``AppConfig(audio={"block_size": 512})`` instead of
        nesting dataclasses by hand, and keeps the section defaults in one
        place.
        """
        for name, class_name in self._SECTIONS.items():
            value = getattr(self, name)
            if isinstance(value, dict):
                klass = globals()[class_name]
                valid = {k: v for k, v in value.items() if k in {f.name for f in fields(klass)}}
                unknown = set(value) - set(valid)
                if unknown:
                    raise ConfigError(
                        f"unknown key(s) in '{name}': {sorted(unknown)}"
                    )
                setattr(self, name, klass(**valid))
        self.validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> "AppConfig":
        """Check cross-section constraints.  Returns self for chaining."""
        if self.audio.block_size <= 0:
            raise ConfigError("audio.block_size must be positive")
        if self.audio.queue_max_blocks <= 0:
            raise ConfigError("audio.queue_max_blocks must be positive")
        if self.audio.sample_rate <= 0:
            raise ConfigError("audio.sample_rate must be positive")
        if self.audio.channels != INTERNAL_CHANNELS:
            # Phase 1 implements a mono processing path only (AGENTS.md 6).
            raise ConfigError(
                f"audio.channels must be {INTERNAL_CHANNELS} for the internal "
                f"format, got {self.audio.channels}"
            )
        if self.event.buffer_required_seconds > self.buffer.seconds:
            raise ConfigError(
                "buffer.seconds must be at least pre_roll + margin "
                f"({self.event.buffer_required_seconds}s), got "
                f"{self.buffer.seconds}s"
            )
        if self.analysis.hop_samples_at(self.sample_rate) >= self.capacity_frames:
            raise ConfigError(
                "analysis hop must be shorter than the ring buffer"
            )
        return self

    # ------------------------------------------------------------------
    # Derived values
    # ------------------------------------------------------------------
    @property
    def sample_rate(self) -> int:
        """Internal sample rate.  The whole pipeline speaks 48 kHz float32."""
        return self.audio.sample_rate

    @property
    def capacity_frames(self) -> int:
        return int(self.buffer.seconds * self.sample_rate)

    @property
    def analysis_frame_frames(self) -> int:
        """Frame length in samples at the internal rate."""
        return self.analysis.frame_samples_at(self.sample_rate)

    @property
    def analysis_hop_frames(self) -> int:
        return self.analysis.hop_samples_at(self.sample_rate)

    @property
    def analysis_frames_per_second(self) -> float:
        """Analysis rate, in frames per second."""
        return self.sample_rate / self.analysis_hop_frames

    @property
    def pre_roll_frames(self) -> int:
        return int(self.event.pre_roll_seconds * self.sample_rate)

    @property
    def post_roll_frames(self) -> int:
        return int(self.event.post_roll_seconds * self.sample_rate)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        if not isinstance(data, dict):
            raise ConfigError("configuration root must be an object")
        kwargs: dict[str, Any] = {}
        for name in cls._SECTIONS:
            raw = data.get(name, {})
            if not isinstance(raw, dict):
                raise ConfigError(f"config section '{name}' must be an object")
            kwargs[name] = raw
        if "output_dir" in data:
            kwargs["output_dir"] = str(data["output_dir"])
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str) -> "AppConfig":
        if not os.path.exists(path):
            return cls().validate()
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
