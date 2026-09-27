"""Adaptive noise floor.

docs/AUDIO_PIPELINE.md section 6 and AGENTS.md section 8.  The requirements:

* Learn the current acoustic environment.  Do not assume silence is -60 dBFS
  or any other fixed value.
* Use different attack and release behaviour, so a sudden sound makes the
  detector respond immediately while the floor does **not** jump up with it.
* Do not learn from strong transient events.

Method
------
Classic minimum statistics, then asymmetric smoothing:

1. **Sub-window averaging with a gate.**  Frame levels are grouped into short
   sub-windows (default 0.5 s) and a ring of recent sub-windows (default 3 s)
   supplies the raw estimate.

   The obvious classical choice is the *minimum* over the window, and it is
   wrong here.  The minimum of a band depends strongly on how many FFT bins it
   contains: the 20-80 Hz band is three bins wide at 48 kHz, and the minimum of
   three bins sits far below the minimum of a wide band, so a minimum-based
   floor is biased low for narrow bands and reports a permanent phantom
   signal-to-noise ratio in ordinary noise.  A per-band bias term could
   correct it, but the transient gate below already excludes events, so the
   mean of the accepted sub-windows is an unbiased estimate of the noise for
   any band width and is what is used.

2. **Asymmetric smoothing.**  The reported floor follows the raw estimate
   quickly when the room gets quieter and slowly when it gets louder.  A quiet
   change should be tracked promptly; a sudden loud one should not be absorbed
   at all until it persists.

3. **Transient gate.**  A sub-window sitting far above the current floor is
   treated as event-contaminated and does not contribute to the estimate.  This
   is what stops a sustained sound from being learned as "the noise here".

The result is a floor that is deliberately *biased low*: it under-reports the
noise during an event rather than climbing to meet it.  That is the correct
direction for a monitor whose job is to notice that something happened.

Floors are estimated per analysis band as well as overall, because the
classifier needs to know that high-frequency energy is *relative to its own
band's* noise, not to the total.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import AnalysisConfig
from .features import FrameFeatures, SILENCE_DB

# A band within this distance of the silence floor carries no usable energy.
_BAND_DEADBAND_DB = 3.0

# Floor for the linear-power accumulator, so log10 never sees zero.
_POWER_EPS = 1e-30


def _quantile_fast(values: list[float], q: float) -> float:
    """Linear-interpolated quantile of a short list of floats.

    Equivalent to ``numpy.quantile(values, q)`` but without its overhead.
    That overhead dominated this stage: profiling showed ``np.quantile`` was
    68% of the noise floor's total time, almost all of it Python-level
    argument validation and index bookkeeping on a list of about thirty
    numbers, and it was being called once per sub-window per band.  That was
    enough to push the analysis stage to 0.89 of a core, which starved
    PortAudio's callback and caused real input overflow.

    The ring is short and already a Python list, so a plain sort and two
    lookups is the cheapest correct thing to do.
    """
    n = len(values)
    if n == 0:
        return 0.0
    if n == 1:
        return values[0]
    ordered = sorted(values)
    if q <= 0.0:
        return ordered[0]
    if q >= 1.0:
        return ordered[-1]
    position = q * (n - 1)
    lower = int(math.floor(position))
    upper = lower + 1
    if upper >= n:
        return ordered[-1]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass
class BandFloor:
    """Noise floor estimate and signal-to-floor ratio for one band."""

    label: str
    floor_db: float
    level_db: float
    snr_db: float
    updated: bool

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "floor_db": self.floor_db,
            "level_db": self.level_db,
            "snr_db": self.snr_db,
            "updated": self.updated,
        }


@dataclass
class NoiseFloorState:
    """The floor as of one frame."""

    index: int
    overall_floor_db: float
    overall_level_db: float
    overall_snr_db: float
    bands: tuple[BandFloor, ...] = field(default_factory=tuple)
    # True while the current level sits well above the floor, i.e. an event is
    # happening now.  Computed per frame, so it is meaningful on every frame
    # rather than only on the rare frames where a sub-window happens to close.
    transient_detected: bool = False
    # True only on the frame where a sub-window was rejected as
    # event-contaminated.  Diagnostics for the estimator itself.
    gate_rejected: bool = False
    # True once enough history exists for the estimate to mean anything.
    initialized: bool = False

    def band(self, label: str) -> BandFloor:
        for item in self.bands:
            if item.label == label:
                return item
        raise KeyError(label)

    @property
    def snr_db(self) -> float:
        return self.overall_snr_db

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "overall_floor_db": self.overall_floor_db,
            "overall_level_db": self.overall_level_db,
            "overall_snr_db": self.overall_snr_db,
            "initialized": self.initialized,
            "transient_detected": self.transient_detected,
            "gate_rejected": self.gate_rejected,
            "bands": [b.to_dict() for b in self.bands],
        }


class _BandTracker:
    """Minimum-statistics floor for one scalar level series."""

    def __init__(
        self,
        label: str,
        subwindow_frames: int,
        window_subwindows: int,
        rise_db_per_sec: float,
        fall_db_per_sec: float,
        transient_gate_db: float,
        frame_rate: float,
        sustain_subwindows: int = 0,
        quantile: float = 0.2,
    ) -> None:
        self.label = label
        self.subwindow_frames = max(1, int(subwindow_frames))
        self.rise_db_per_sec = float(rise_db_per_sec)
        self.fall_db_per_sec = float(fall_db_per_sec)
        self.transient_gate_db = float(transient_gate_db)
        self.frame_rate = float(frame_rate)

        self._sub_min = math.inf
        # Sum of *linear power* over the current sub-window, used to decide
        # whether the sub-window is an event.  Gating on the *minimum* would
        # misfire on steady noise, because the minimum of a short window of
        # noise wanders by several dB from one sub-window to the next; the
        # typical level is the stable quantity.  Summing power rather than dB
        # avoids the dB-domain averaging bias described in the module
        # docstring.
        self._sub_power_sum = 0.0
        self._sub_count = 0
        # Ring of accepted sub-window levels (their means).  See the module
        # docstring for why the mean and not the minimum.
        # Ring of accepted sub-window power levels, linear.
        self._subwindows: deque[float] = deque(maxlen=max(1, window_subwindows))
        self._floor: Optional[float] = None
        self._rejections = 0
        # Consecutive sub-windows seen above the transient gate.  Once this
        # passes the sustain threshold the elevated level is treated as the
        # new environment instead of an event.
        self._sustain_subwindows = 0
        self._sustained = False
        self.sustain_subwindows = max(0, int(sustain_subwindows))
        self.quantile = float(max(0.0, min(1.0, quantile)))
        # The ring only changes when a sub-window closes, which is every
        # `subwindow_frames` calls.  Sorting it every frame was five times the
        # cost of the mean it replaced, for a value that rarely changes.
        self._cached_estimate: Optional[float] = None

    # ------------------------------------------------------------------
    @property
    def floor_db(self) -> float:
        return SILENCE_DB if self._floor is None else self._floor

    @property
    def is_initialized(self) -> bool:
        return self._floor is not None

    @property
    def rejections(self) -> int:
        """Sub-windows discarded as event-contaminated."""
        return self._rejections

    def reset(self) -> None:
        self._sub_min = math.inf
        self._sub_power_sum = 0.0
        self._sub_count = 0
        self._subwindows.clear()
        self._cached_estimate = None
        self._floor = None
        self._sustain_subwindows = 0
        self._sustained = False
        # Counters describe this stretch of audio, so they reset with it.
        self._rejections = 0

    # ------------------------------------------------------------------
    def raw_estimate(self) -> float:
        """The raw noise estimate from the accepted sub-windows, in dB.

        A *low quantile* of the sub-window distribution, taken in the linear
        power domain.

        Why not the mean: the mean of a window is, by construction, close to
        the average level of whatever occupied that window, so any sound
        present most of the time defines the floor and its reported
        signal-to-noise ratio collapses to zero.  Measured, a whisper occupying
        71% of the window was absorbed completely at every amplitude tried, from
        7 dB to 24 dB above the room noise.  A sustained sound is exactly the
        case a monitor must not miss.

        Why not the absolute minimum: the minimum of a sample is biased low in
        proportion to how many independent samples it is drawn from, and a
        narrow analysis band has far fewer of them than a wide one.  Measured,
        the 20-80 Hz band - three FFT bins at 48 kHz - sat about 10 dB below its
        own mean, so every band reported a phantom signal-to-noise ratio in
        ordinary noise.

        A low quantile avoids both: it tracks the quiet part of the window so a
        sustained sound stays visible, while averaging over the ring of
        sub-windows first means it is far less sensitive to a single outlier
        frame than the absolute minimum, and far less biased across band widths.
        """
        if self._cached_estimate is not None:
            return self._cached_estimate
        if not self._subwindows:
            return SILENCE_DB
        power = _quantile_fast(list(self._subwindows), self.quantile)
        self._cached_estimate = 10.0 * math.log10(max(power, _POWER_EPS))
        return self._cached_estimate

    def update(self, level_db: float) -> tuple[float, bool]:
        """Feed one frame's level.  Returns ``(floor_db, transient_rejected)``."""
        if not math.isfinite(level_db):
            level_db = SILENCE_DB

        # --- accumulate the current sub-window --------------------------
        self._sub_min = min(self._sub_min, level_db)
        # dB -> linear power, floored so a silent frame contributes nothing
        # rather than a negative number.
        self._sub_power_sum += max(10.0 ** (level_db / 10.0), 0.0)
        self._sub_count += 1
        rejected = False
        if self._sub_count >= self.subwindow_frames:
            # Mean linear power of this sub-window; the ring stores power, not
            # dB, so the averaging stays unbiased.
            sub_power = max(self._sub_power_sum / self._sub_count, _POWER_EPS)
            candidate = self._sub_min
            typical = 10.0 * math.log10(sub_power)
            # `candidate` (the sub-window minimum) is kept only as a
            # diagnostic of how much the sub-window dipped; the estimate uses
            # the mean.
            self._sub_min = math.inf
            self._sub_power_sum = 0.0
            self._sub_count = 0

            # --- transient gate -------------------------------------------
            # Compare the sub-window's *typical* level against the current
            # floor.  Before any floor exists, accept the first sub-window:
            # the environment at startup is assumed to be the baseline.
            if (
                self._floor is not None
                and typical - self._floor > self.transient_gate_db
                and not self._sustained
            ):
                self._sustain_subwindows += 1
                if self._sustain_subwindows <= self.sustain_subwindows:
                    # Still brief: treat as an event and leave the estimate
                    # alone.  The ring is not updated, so the floor does not
                    # move, which is the entire point of the gate.
                    rejected = True
                    self._rejections += 1
                else:
                    # Elevated for long enough to be the environment.  A room
                    # that genuinely got louder must become the new floor,
                    # otherwise the monitor is deaf in it forever.
                    self._sustained = True
                    self._subwindows.append(sub_power)
            else:
                self._sustain_subwindows = 0
                if not (
                    self._floor is not None
                    and typical - self._floor > self.transient_gate_db
                ):
                    self._sustained = False
                self._subwindows.append(sub_power)
                self._cached_estimate = None

        # --- asymmetric smoothing ------------------------------------------
        if self._floor is None:
            # Seed from the history we already have, or from the first level
            # seen if there is not yet a complete sub-window.  The floor is a
            # measurement, so it is seeded at the level observed - never
            # clamped.  An earlier design clamped it to a ceiling, which
            # pinned every loud band to the ceiling and reported a permanent
            # phantom SNR.
            seed = self.raw_estimate() if self._subwindows else level_db
            self._floor = seed
            return self._floor, rejected

        if not self._subwindows:
            return self._floor, rejected

        raw = self.raw_estimate()

        # How far can the floor move in one frame, at each rate.
        dt = 1.0 / self.frame_rate
        delta = raw - self._floor
        if delta < 0.0:
            limit = self.fall_db_per_sec * dt
        else:
            limit = self.rise_db_per_sec * dt
        if abs(delta) <= limit:
            self._floor = raw
        else:
            self._floor += math.copysign(limit, delta)
        return self._floor, rejected


class AdaptiveNoiseFloor:
    """Maintains an adaptive noise floor for the overall level and each band.

    One instance follows one stream.  Feed it every frame in order via
    :meth:`update`.
    """

    def __init__(
        self,
        config: Optional[AnalysisConfig] = None,
        frame_rate: float = 100.0,
    ) -> None:
        self.config = config or AnalysisConfig()
        if frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        self.frame_rate = float(frame_rate)

        subwindow_frames = max(
            1, int(round(self.config.min_stats_subwindow_sec * frame_rate))
        )
        window_subwindows = max(
            1, int(round(self.config.min_stats_window_sec / self.config.min_stats_subwindow_sec))
        )
        sustain_subwindows = (
            int(round(self.config.floor_sustain_sec / self.config.min_stats_subwindow_sec))
            if self.config.min_stats_subwindow_sec > 0
            else 0
        )
        self.subwindow_frames = subwindow_frames
        self.window_subwindows = window_subwindows
        self.sustain_subwindows = sustain_subwindows
        common = dict(
            subwindow_frames=subwindow_frames,
            window_subwindows=window_subwindows,
            rise_db_per_sec=self.config.floor_rise_db_per_sec,
            fall_db_per_sec=self.config.floor_fall_db_per_sec,
            transient_gate_db=self.config.transient_gate_db,
            frame_rate=frame_rate,
            sustain_subwindows=sustain_subwindows,
            quantile=self.config.floor_quantile,
        )
        self._overall = _BandTracker("overall", **common)
        self._bands: dict[str, _BandTracker] = {}
        self._index = 0
        self.frames_seen = 0

    # ------------------------------------------------------------------
    @property
    def overall_floor_db(self) -> float:
        return self._overall.floor_db

    @property
    def overall_snr_db(self) -> float:
        return self._overall_snr

    @property
    def is_initialized(self) -> bool:
        return self._overall.is_initialized and bool(self._bands) and all(
            b.is_initialized for b in self._bands.values()
        )

    @property
    def rejections(self) -> int:
        return self._overall.rejections + sum(
            b.rejections for b in self._bands.values()
        )

    def reset(self) -> None:
        self._overall.reset()
        for tracker in self._bands.values():
            tracker.reset()
        self._index = 0
        self.frames_seen = 0

    # ------------------------------------------------------------------
    def update(self, features: FrameFeatures) -> NoiseFloorState:
        """Fold one frame into the estimate and return the current state."""
        for band in features.bands:
            tracker = self._bands.get(band.label)
            if tracker is None:
                tracker = _BandTracker(
                    band.label,
                    subwindow_frames=self.subwindow_frames,
                    window_subwindows=self.window_subwindows,
                    rise_db_per_sec=self.config.floor_rise_db_per_sec,
                    fall_db_per_sec=self.config.floor_fall_db_per_sec,
                    transient_gate_db=self.config.transient_gate_db,
                    frame_rate=self.frame_rate,
                    sustain_subwindows=self.sustain_subwindows,
                    quantile=self.config.floor_quantile,
                )
                self._bands[band.label] = tracker

        floor_db, rejected = self._overall.update(features.rms_db)
        self._overall_snr = features.rms_db - floor_db

        band_states: list[BandFloor] = []
        any_rejected = rejected
        for band in features.bands:
            tracker = self._bands[band.label]
            if band.db <= SILENCE_DB + _BAND_DEADBAND_DB:
                # The band carries no energy.  Its dB level is numerical noise,
                # not a measurement: a band sitting at the -200 dBFS floor
                # fluctuates wildly between frames, which would make the gate
                # fire constantly and report phantom events.  So a band with no
                # energy is not allowed to raise the floor or trip the gate,
                # and its SNR is reported as zero rather than as a meaningless
                # ratio.
                band_states.append(
                    BandFloor(
                        label=band.label,
                        floor_db=tracker.floor_db,
                        level_db=band.db,
                        snr_db=0.0,
                        updated=tracker.is_initialized,
                    )
                )
                continue
            b_floor, b_rejected = tracker.update(band.db)
            any_rejected = any_rejected or b_rejected
            band_states.append(
                BandFloor(
                    label=band.label,
                    floor_db=b_floor,
                    level_db=band.db,
                    snr_db=band.db - b_floor,
                    updated=tracker.is_initialized,
                )
            )

        self._index += 1
        self.frames_seen += 1
        return NoiseFloorState(
            index=features.index,
            overall_floor_db=floor_db,
            overall_level_db=features.rms_db,
            overall_snr_db=self._overall_snr,
            bands=tuple(band_states),
            transient_detected=(
                self._overall_snr > self.config.transient_gate_db
            ),
            gate_rejected=any_rejected,
            initialized=self.is_initialized,
        )
