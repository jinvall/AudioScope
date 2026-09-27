"""Candidate detection.

docs/AUDIO_PIPELINE.md section 7 lists what may create a candidate: energy
change, spectral flux, unusual band behaviour, broadband transients, modulation,
persistent low-level energy, harmonic change, unusual spectral shape.

The guiding constraint is docs/DETECTION_AND_CLASSIFICATION.md section 2:
detection should be deliberately broad, because it is better to record an
*unknown* event than to miss a quiet one.  So this stage answers only "did
something happen?", never "what was it?".  Deciding what it was is the
classifier's job, and it is allowed to answer "unknown".

Every indicator is a *relative* measure against the adaptive noise floor, never
a fixed dBFS value (AGENTS.md section 8).  A sound 6 dB above a -20 dBFS room is
an event; the same sound in a -60 dBFS room is not.

Indicators are combinable: the activation is the strongest single indicator, not
a product.  A product would let one weak indicator veto the rest, which is the
opposite of the broadness wanted here.  Which indicators contributed is carried
along so the classifier and the UI can show *why* something was flagged rather
than just that it was.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import DetectionConfig
from ..analysis.features import FrameFeatures
from ..analysis.noise_floor import NoiseFloorState

# Band used as the "low frequency" indicator: footstep thump, movement, cloth.
LOW_BAND = "80-250"
# Band used as the "high frequency" indicator: whisper, sibilance, brushing.
HIGH_BAND = "2000-4000"
# Middle band, 500-2000 Hz.  Named for its position in the spectrum rather than
# for a sound class: the detector core must not encode assumptions about which
# sounds occupy which part of the spectrum.
MID_BAND = "500-2000"


def _clamp01(value: float) -> float:
    if value != value:  # NaN
        return 0.0
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def _ramp(value: float, low: float, high: float) -> float:
    """Map ``value`` onto 0..1 between ``low`` and ``high``."""
    if high <= low:
        return 0.0
    return _clamp01((value - low) / (high - low))


@dataclass
class Candidate:
    """One frame's assessment of whether something is happening."""

    index: int
    start_sample: int
    audio_time: float
    # 0..1, the strongest single indicator.
    activation: float
    # Names of the indicators that contributed, strongest first.
    indicators: tuple[str, ...]
    # Per-indicator values, for the classifier and for explaining a detection.
    values: dict[str, float] = field(default_factory=dict)
    # SNR above the adaptive floor at this frame.
    snr_db: float = 0.0
    # Envelope modulation depth over the modulation window.
    modulation: float = 0.0
    # Fraction of samples at or near full scale; > 0 means clipping.
    clipping_ratio: float = 0.0
    # Dominant frequency and tonal-ness, for interference analysis.
    dominant_frequency_hz: float = 0.0
    spectral_peakiness: float = 0.0
    # Current flux divided by its running median; 1.0 means "as usual".
    flux_ratio: Optional[float] = None
    crest_factor: float = 0.0

    @property
    def is_candidate(self) -> bool:
        return self.activation > 0.0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start_sample": self.start_sample,
            "audio_time": self.audio_time,
            "activation": self.activation,
            "indicators": list(self.indicators),
            "values": dict(self.values),
            "snr_db": self.snr_db,
            "modulation": self.modulation,
            "clipping_ratio": self.clipping_ratio,
            "dominant_frequency_hz": self.dominant_frequency_hz,
            "spectral_peakiness": self.spectral_peakiness,
            "flux_ratio": self.flux_ratio,
            "crest_factor": self.crest_factor,
        }


class CandidateDetector:
    """Turns a stream of analysis results into per-frame candidates.

    Stateful: it keeps a short rolling RMS history for the modulation measure.
    Feed it every analysis result from one stream, in order.
    """

    def __init__(
        self,
        config: Optional[DetectionConfig] = None,
        frame_rate: float = 100.0,
    ) -> None:
        self.config = config or DetectionConfig()
        if frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        self.frame_rate = float(frame_rate)
        window = max(
            2, int(round(self.config.modulation_window * frame_rate))
        )
        self._rms_history: deque[float] = deque(maxlen=window)
        self._window = window
        # Running history for the indicators that have no meaningful absolute
        # scale.  Flux for background noise is large by construction and crest
        # for noise is around 3.6, so both are only interpretable against their
        # own recent behaviour.
        baseline = max(
            2, int(round(self.config.baseline_window * frame_rate))
        )
        self._flux_history: deque[float] = deque(maxlen=baseline)
        self._crest_history: deque[float] = deque(maxlen=baseline)
        self.frames_seen = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._rms_history.clear()
        self._flux_history.clear()
        self._crest_history.clear()
        self.frames_seen = 0

    @property
    def history_length(self) -> int:
        return len(self._rms_history)

    # ------------------------------------------------------------------
    def update(
        self,
        features: FrameFeatures,
        floor: NoiseFloorState,
        start_sample: int,
    ) -> Candidate:
        """Assess one frame."""
        config = self.config
        values: dict[str, float] = {}

        # --- level relative to the floor -------------------------------
        snr = floor.overall_snr_db
        # A sustained level counts too, not just a sudden one: docs section 7
        # lists "persistent low-level energy" as a valid candidate source.
        level = _ramp(snr, config.minimum_event_snr_db, config.snr_full_db)
        if level > 0.0:
            values["level"] = level

        # --- spectral flux, relative to its own history -------------------
        # An absolute threshold is useless here: uncorrelated background noise
        # already produces a flux of about 0.38, so any constant would flag the
        # room itself.  What means something is a *jump* above the recent
        # median, which is what marks an onset.
        flux = features.spectral_flux
        flux_ratio: Optional[float] = None
        if flux is not None:
            if len(self._flux_history) >= 8:
                baseline = float(np.median(self._flux_history))
                if baseline > 1e-6:
                    flux_ratio = flux / baseline
                    value = _ramp(
                        flux_ratio, config.flux_ratio_low, config.flux_ratio_full
                    )
                    if value > 0.0:
                        values["flux"] = value
            # Recorded after use, so the baseline excludes the current frame.
            self._flux_history.append(flux)

        # --- per-band energy against that band's own floor --------------
        # Using each band's own floor is what makes "unusual frequency-band
        # change" meaningful in a room with a loud fan.
        low_snr = _band_snr(floor, LOW_BAND)
        if low_snr is not None:
            value = _ramp(low_snr, config.minimum_event_snr_db, config.low_band_full_db)
            if value > 0.0:
                values["low_band"] = value

        high_snr = _band_snr(floor, HIGH_BAND)
        if high_snr is not None:
            value = _ramp(high_snr, config.minimum_event_snr_db, config.high_band_full_db)
            if value > 0.0:
                values["high_band"] = value

        # --- impulsiveness ---------------------------------------------
        # Crest above the noise ceiling, which is what distinguishes a sharp
        # click or a tap from steady noise.  It is deliberately NOT evidence of
        # a footstep: a 40 ms frame holds only a few cycles of a decaying low
        # thump, so footsteps measure *below* the surrounding noise. See the
        # DetectionConfig comments.
        crest = features.crest_factor
        if crest > config.crest_low:
            value = _ramp(crest, config.crest_low, config.crest_full)
            if value > 0.0:
                values["impulsive"] = value
        if len(self._crest_history) >= 8:
            self._crest_history.append(crest)

        # --- modulation ------------------------------------------------
        self._rms_history.append(features.rms)
        modulation = self._modulation_depth()
        if modulation > 0.0:
            value = _ramp(modulation, config.modulation_low, config.modulation_full)
            if value > 0.0:
                values["modulation"] = value

        # --- clipping is an event in its own right ----------------------
        # The sample histogram is not a Phase 2 feature, so it is derived here
        # from the peak and RMS: a signal at or beyond full scale with a flat
        # top has crest far below what a transient of the same peak would have.
        # This is a coarse proxy, documented as such.
        clipping = self._clipping_ratio(features)

        activation = max(values.values()) if values else 0.0
        # Clipping above the minimum event level is itself a candidate even when
        # the level is unremarkable, because a clipped signal means the input
        # stage is saturating.
        if clipping > 0.0 and snr >= config.minimum_event_snr_db:
            activation = max(activation, _ramp(clipping, 0.001, 0.05))

        ordered = tuple(
            sorted(values, key=lambda k: values[k], reverse=True)
        )
        self.frames_seen += 1
        return Candidate(
            index=features.index,
            start_sample=int(start_sample),
            audio_time=int(start_sample) / features.sample_rate,
            activation=activation,
            indicators=ordered,
            values=values,
            snr_db=snr,
            modulation=modulation,
            clipping_ratio=clipping,
            dominant_frequency_hz=features.dominant_frequency_hz,
            spectral_peakiness=features.spectral_peakiness,
            flux_ratio=flux_ratio,
            crest_factor=crest,
        )

    # ------------------------------------------------------------------
    def _modulation_depth(self) -> float:
        """Envelope modulation depth over the rolling window, in 0..1.

        Speech is modulated at roughly 2-8 Hz, so a 1 s window spans several
        cycles.  Depth is the peak-to-trough spread of the RMS envelope relative
        to the envelope's own *peak*.

        Normalising by the peak rather than the mean is deliberate.  The
        textbook modulation index divides by the mean, which is unbounded: a
        single loud frame in an otherwise quiet window pushes it past 10, the
        indicator saturates, and every active sound then scores identically.
        Measured on this system, dividing by the peak gives 0.07 for quiet
        noise and about 0.98 for either a footstep sequence or modulated speech -
        bounded, and honest about the fact that *both* are modulated.  Telling
        those two apart is the job of the presence measure, not this one.

        Returns 0 until the window is full: a modulation figure computed from
        two samples would be meaningless.
        """
        if len(self._rms_history) < max(4, self._window // 2):
            return 0.0
        values = np.fromiter(self._rms_history, dtype=np.float64)
        peak = float(values.max())
        if peak <= 1e-9:
            return 0.0
        return float((peak - float(values.min())) / peak)

    def _clipping_ratio(self, features: FrameFeatures) -> float:
        """Coarse clipping estimate from peak and crest factor.

        The exact measure is "fraction of samples at full scale", which needs
        the raw frame and belongs to the analysis stage.  This is derived from
        what the features expose, so it is only a proxy: a signal whose peak is
        within 0.1 dB of full scale *and* whose crest factor is low enough that
        the peak cannot be a lone transient is reported as clipping.

        Returns 0 when the peak is below 0.999, since nothing is clipped then.
        """
        if features.peak < 0.999:
            return 0.0
        # A true transient at full scale still has a high crest factor.
        if features.crest_factor >= 4.0:
            return 0.0
        # Map crest 4.0 -> 0.0 and crest 1.2 -> 0.04.
        return float(_clamp01((4.0 - features.crest_factor) / 2.8) * 0.04)


def _band_snr(floor: NoiseFloorState, label: str) -> Optional[float]:
    try:
        return floor.band(label).snr_db
    except KeyError:
        return None
