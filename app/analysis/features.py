"""Per-frame acoustic features.

Implements the feature set from tasks/PHASE_02_ANALYSIS.md and
docs/AUDIO_PIPELINE.md section 5:

    RMS, peak, crest factor, zero-crossing rate, spectral centroid,
    spectral bandwidth, spectral rolloff, spectral flatness, spectral flux,
    band energies

plus DC offset, which AGENTS.md section 12 needs for interference analysis.

Cost matters: this runs continuously on every frame, so everything is
precomputed where possible (window, band bin masks) and the per-frame work is
one real FFT plus a handful of vector operations.  No allocation per call
beyond the returned arrays.

Units and conventions
---------------------
* Levels are linear float in [-1, 1]; ``*_db`` fields are ``20*log10`` of a
  linear magnitude and use :data:`SILENCE_DB` rather than -inf, so a silent
  frame still produces a usable number instead of an exception or a NaN.
* Spectral features are in Hz, computed against the true frequency axis.
* Band energies are reported both linear (summed power) and in dB.
* Nothing here is a confidence value.  These are measurements, not opinions
  (AGENTS.md section 2.4, docs/DETECTION_AND_CLASSIFICATION.md section 8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .frames import ANALYSIS_BANDS, BAND_LABELS, window_function

# Smallest magnitude treated as non-zero when converting to dB.  -200 dBFS is
# far below anything a real ADC produces and keeps the arithmetic finite.
SILENCE_AMPLITUDE = 1e-10
SILENCE_DB = -200.0
_EPS = 1e-12

# Relative deadband for zero-crossing detection.  Without it, numerical dust
# around a constant (DC) signal reads as a stream of crossings.
_ZCR_DEADBAND = 1e-4


def amplitude_to_db(value: float | np.ndarray) -> float | np.ndarray:
    """Convert a linear magnitude(s) to dBFS, flooring at SILENCE_DB."""
    if isinstance(value, (float, int, np.floating, np.integer)):
        # Scalar path. This is called about twenty times per analysis frame -
        # once per band, plus density and peak - and routing each one through
        # numpy costs roughly twenty times more than the arithmetic itself.
        return 20.0 * math.log10(max(abs(float(value)), SILENCE_AMPLITUDE))
    return 20.0 * np.log10(
        np.maximum(np.abs(value), SILENCE_AMPLITUDE)
    )


def db_to_amplitude(db: float) -> float:
    return float(10.0 ** (db / 20.0))


@dataclass(frozen=True)
class BandEnergy:
    """Energy in one analysis band.

    ``linear``/``db`` are the band's **total** energy, so a wide band naturally
    sums more than a narrow one and the two are not directly comparable.
    ``density_*`` divide by the band width to give power per hertz, which is
    what makes "is there more high-frequency energy than usual?" answerable.
    Both are reported rather than one being silently substituted, because the
    noise floor already adapts per band and only the density form supports
    cross-band comparison.
    """

    label: str
    low_hz: float
    high_hz: float
    linear: float
    db: float
    density_linear: float
    density_db: float

    @property
    def width_hz(self) -> float:
        return self.high_hz - self.low_hz

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "low_hz": self.low_hz,
            "high_hz": self.high_hz,
            "width_hz": self.width_hz,
            "linear": self.linear,
            "db": self.db,
            "density_linear": self.density_linear,
            "density_db": self.density_db,
        }


@dataclass(frozen=True)
class FrameFeatures:
    """Measurements for one analysis frame.

    ``spectral_flux`` and ``band_ratios`` are absent on the very first frame
    because both need history; they are ``None`` rather than a fabricated 0.0 so
    a consumer can tell "no flux" from "flux of zero".
    """

    # Timing
    index: int
    start_sample: int
    sample_rate: int
    hop_samples: int

    # Level
    rms: float
    peak: float
    crest_factor: float
    zero_crossing_rate: float
    dc_offset: float
    rms_db: float
    peak_db: float

    # Spectral shape
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    spectral_rolloff_hz: float
    spectral_flatness: float
    spectral_flux: Optional[float]
    # Strongest bin's frequency. Interference analysis needs this to recognise
    # 50/60 Hz hum and its harmonics (AGENTS.md section 12, docs section 6).
    dominant_frequency_hz: float
    # Strongest bin relative to the median bin. Near 1 is broadband, large for a
    # narrow peak. Distinguishes a hum from speech at the same level.
    spectral_peakiness: float
    centroid_ratio: Optional[float] = None

    # Bands
    bands: tuple[BandEnergy, ...] = field(default_factory=tuple)

    # Convenience
    #: Peakiness above which a frame counts as tonal.
    #:
    #: Measured on 1920-sample frames at 48 kHz: white noise scores about 13
    #: (the largest of ~961 exponential bins sits near ``ln(961)`` times the
    #: mean, and the median is 0.69 of it), a three-tone harmonic stack about
    #: 213, and a single pure tone 541-640. 100 therefore separates noise from
    #: harmonic content with room on both sides.  It is a convenience flag;
    #: the classifier uses the continuous value rather than this boolean.
    TONAL_PEAKINESS = 100.0

    @property
    def is_tonal(self) -> bool:
        """A narrow, stable peak: hum, a motor, a ringing surface.

        Note this is also true of a clipped tone and of LF-dominated noise,
        which both saturate the ratio.  Clipping is detected separately from
        the sample histogram, so a distorted hum is not mistaken for a clean
        one.
        """
        return self.spectral_peakiness >= self.TONAL_PEAKINESS

    @property
    def band_dict(self) -> dict[str, float]:
        return {b.label: b.db for b in self.bands}

    @property
    def band_linear(self) -> dict[str, float]:
        return {b.label: b.linear for b in self.bands}

    def band_db(self, label: str) -> float:
        for band in self.bands:
            if band.label == label:
                return band.db
        raise KeyError(label)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start_sample": self.start_sample,
            "sample_rate": self.sample_rate,
            "hop_samples": self.hop_samples,
            "rms": self.rms,
            "peak": self.peak,
            "crest_factor": self.crest_factor,
            "zero_crossing_rate": self.zero_crossing_rate,
            "dc_offset": self.dc_offset,
            "rms_db": self.rms_db,
            "peak_db": self.peak_db,
            "spectral_centroid_hz": self.spectral_centroid_hz,
            "spectral_bandwidth_hz": self.spectral_bandwidth_hz,
            "spectral_rolloff_hz": self.spectral_rolloff_hz,
            "spectral_flatness": self.spectral_flatness,
            "spectral_flux": self.spectral_flux,
            "dominant_frequency_hz": self.dominant_frequency_hz,
            "spectral_peakiness": self.spectral_peakiness,
            "is_tonal": self.is_tonal,
            "bands": [b.to_dict() for b in self.bands],
        }


class FeatureExtractor:
    """Computes :class:`FrameFeatures` for successive frames.

    The extractor is stateful because spectral flux compares each frame's
    magnitude spectrum with the previous one.  Feed it consecutive frames from a
    single stream, in order.

    Parameters
    ----------
    sample_rate:
        Frame rate, used to build the frequency axis.
    bands:
        Band edges in Hz.  Bands above Nyquist are still reported, with zero
        energy, so the band set does not change shape with the sample rate.
    rolloff_percentile:
        Fraction of total spectral energy below which the rolloff lies.
    window:
        Window applied internally to the copy used for the spectrum.  Time
        domain features are always measured on the raw samples.
    """

    def __init__(
        self,
        sample_rate: int,
        bands: tuple[tuple[float, float], ...] = ANALYSIS_BANDS,
        rolloff_percentile: float = 0.85,
        flux_power: bool = True,
        window: Optional[str] = "hann",
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 0.0 < rolloff_percentile <= 1.0:
            raise ValueError("rolloff_percentile must be in (0, 1]")

        self.sample_rate = int(sample_rate)
        self.bands = tuple((float(lo), float(hi)) for lo, hi in bands)
        self.rolloff_percentile = float(rolloff_percentile)
        self.flux_power = bool(flux_power)
        self.window_name = window

        self._nyquist = self.sample_rate / 2.0
        self._frame_samples: Optional[int] = None
        self._window: Optional[np.ndarray] = None
        self._freqs: Optional[np.ndarray] = None
        self._bin_masks: Optional[list[np.ndarray]] = None
        self._prev_magnitude: Optional[np.ndarray] = None
        self._last_magnitude: Optional[np.ndarray] = None
        self._index = 0

    @property
    def last_magnitude(self) -> Optional[np.ndarray]:
        """Magnitude spectrum of the most recent frame, or None.

        Reused by the spectrogram instead of recomputing the transform.
        """
        return self._last_magnitude

    # ------------------------------------------------------------------
    def _prepare(self, n: int) -> None:
        """Build the frequency axis and band masks once per frame length."""
        if self._frame_samples == n and self._freqs is not None:
            return
        self._frame_samples = n
        self._window = (
            np.ones(n, dtype=np.float64)
            if self.window_name in (None, "none", "rect")
            else window_function(self.window_name, n)
        )
        self._freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        # Band edges are contiguous in frequency, so each band is a contiguous
        # slice of the rfft output.  Resolving them to integer ranges once means
        # the per-frame cost is nine slice sums taken from a single cumulative
        # sum, instead of nine boolean-mask gathers.
        self._band_slices: list[slice] = []
        for lo, hi in self.bands:
            start = int(np.searchsorted(self._freqs, lo, side="left"))
            stop = int(np.searchsorted(self._freqs, hi, side="left"))
            self._band_slices.append(slice(start, max(stop, start)))
        self._prev_magnitude = None

    @property
    def frequencies(self) -> np.ndarray:
        if self._freqs is None:
            raise RuntimeError(
                "no frame has been analysed yet; call analyse() first"
            )
        return self._freqs

    @property
    def band_edges(self) -> tuple[tuple[float, float], ...]:
        return self.bands

    def reset(self) -> None:
        """Forget the previous frame, so the next frame has no flux."""
        self._prev_magnitude = None
        self._index = 0

    # ------------------------------------------------------------------
    def analyse(self, frame: np.ndarray, start_sample: int = 0) -> FrameFeatures:
        """Compute features for one **unwindowed** frame.

        The window is applied internally, and only to the copy used for the
        spectrum.  Time-domain features (RMS, peak, crest, ZCR, DC) must be
        measured on the raw samples: a Hann window scales a full-scale sine by
        about 0.5, which would halve the reported level and double the crest
        factor.  Doing the windowing here rather than in the frame extractor
        removes the whole class of mistake.
        """
        data = np.asarray(frame, dtype=np.float32).ravel()
        n = data.size
        if n == 0:
            raise ValueError("frame must not be empty")

        self._prepare(n)
        raw = data.astype(np.float64, copy=False)
        # Windowed copy, used only for the spectrum below.
        work = raw * self._window

        # ---- time-domain (unwindowed) ------------------------------------
        peak = float(np.max(np.abs(raw)))
        # One BLAS call, no temporary of n floats.
        rms = float(np.sqrt(np.dot(raw, raw) / n))
        # Crest factor: peak-to-average ratio.  High for impulsive sounds
        # (clicks, taps), low for steady noise.  Undefined for pure silence,
        # so it is capped rather than allowed to explode.
        if rms > _EPS:
            crest = min(peak / rms, 1e4)
        else:
            crest = 0.0
        dc_offset = float(np.mean(raw))
        # Zero-crossing rate on the mean-removed signal, so a DC-biased input
        # does not report a meaningless crossing rate.  Values inside a small
        # deadband relative to the peak are treated as zero, otherwise a
        # constant signal reports a stream of crossings from rounding dust and
        # a genuinely silent frame does the same.
        if n > 1 and peak > _EPS:
            centred = raw - dc_offset
            # A transition is a change of sign between consecutive *significant*
            # samples, so the samples that fall inside the deadband are skipped
            # rather than treated as crossings of their own.
            #
            # Note it is the sample *at* a zero crossing that is insignificant,
            # so requiring both neighbours to be significant would discard every
            # real crossing.  Collecting the significant indices and comparing
            # those is the correct form, and costs one compaction rather than
            # the three array allocations the earlier version made.
            significant = np.flatnonzero(
                np.abs(centred) > peak * _ZCR_DEADBAND
            )
            if significant.size > 1:
                negative = np.signbit(centred[significant])
                changes = int(
                    np.count_nonzero(negative[1:] != negative[:-1])
                )
            else:
                changes = 0
            zcr = float(changes) / float(n - 1)
        else:
            zcr = 0.0

        # ---- spectral ----------------------------------------------------
        spectrum = np.fft.rfft(work)
        power = (spectrum.real ** 2) + (spectrum.imag ** 2)
        freqs = self._freqs

        total_power = float(power.sum())
        if total_power > _EPS:
            centroid = float(np.sum(freqs * power) / total_power)
            variance = float(
                np.sum(((freqs - centroid) ** 2) * power) / total_power
            )
            bandwidth = float(math.sqrt(max(variance, 0.0)))
        else:
            centroid = 0.0
            bandwidth = 0.0

        # Rolloff: lowest frequency below which `rolloff_percentile` of the
        # energy lies.  Normalising by the total makes it independent of level.
        if total_power > _EPS:
            cumulative = np.cumsum(power)
            index = int(
                np.searchsorted(cumulative, self.rolloff_percentile * total_power)
            )
            index = min(index, freqs.size - 1)
            rolloff = float(freqs[index])
        else:
            rolloff = 0.0

        # Flatness: geometric mean over arithmetic mean of the power spectrum.
        # ~1 for broadband noise, near 0 for a pure tone.  Silence is defined
        # as maximally flat, which is the honest answer for an empty spectrum
        # only up to the floor; the energy check keeps it from claiming that.
        mean_power = total_power / power.size
        if mean_power > _EPS:
            # Clamping rather than masking: boolean indexing here allocated a
            # compacted copy of the whole spectrum every frame.
            geometric = float(np.exp(np.mean(np.log(np.maximum(power, _EPS)))))
            flatness = float(min(geometric / mean_power, 1.0))
        else:
            flatness = 0.0

        # Flux: half-rectified difference against the previous frame.  Only
        # positive change counts, so a decaying sound is not a "change".
        magnitude = np.sqrt(power)
        # Kept so a caller that also wants a display spectrum does not have to
        # run a second FFT over the same frame.  The spectrogram used to do
        # exactly that, doubling the transform cost of the whole stage.
        self._last_magnitude = magnitude
        if self.flux_power:
            current = power
        else:
            current = magnitude
        flux: Optional[float] = None
        if self._prev_magnitude is not None and self._prev_magnitude.size == current.size:
            diff = current - self._prev_magnitude
            flux = float(np.sum(np.maximum(diff, 0.0)) / (total_power + _EPS))
        self._prev_magnitude = current

        # ---- peak structure ------------------------------------------------
        # Dominant bin and how far it stands above the median bin. Both are
        # taken from the same spectrum, so this costs one argmax and one
        # median. The dominant bin is the loudest bin from 20 Hz upward, so
        # low-frequency rumble and DC do not masquerade as the tone.
        if total_power > _EPS:
            search_from = int(np.searchsorted(freqs, 20.0, side="left"))
            if search_from >= power.size:
                search_from = 0
            peak_index = search_from + int(np.argmax(power[search_from:]))
            dominant = float(freqs[peak_index])
            median_power = float(np.median(power))
            if median_power > _EPS:
                peakiness = float(power[peak_index] / median_power)
            else:
                peakiness = float(power[peak_index] / (total_power / power.size))
            peakiness = min(peakiness, 1e6)
        else:
            dominant = 0.0
            peakiness = 0.0

        # ---- bands ---------------------------------------------------------
        # One cumulative sum serves every band.
        cumulative = np.cumsum(power)
        band_values: list[BandEnergy] = []
        for (lo, hi), band_slice in zip(self.bands, self._band_slices):
            if band_slice.stop > band_slice.start:
                energy = float(
                    cumulative[band_slice.stop - 1]
                    - (cumulative[band_slice.start - 1]
                       if band_slice.start > 0 else 0.0)
                )
                if energy < 0.0:
                    # Cancellation between large bins; clamp rather than
                    # report a negative energy.
                    energy = 0.0
            else:
                # A band above Nyquist, or narrower than one FFT bin.  Report
                # zero rather than borrowing energy from a neighbour.
                energy = 0.0
            width = max(hi - lo, 1.0)
            density = energy / width
            band_values.append(
                BandEnergy(
                    label=f"{lo:g}-{hi:g}",
                    low_hz=lo,
                    high_hz=hi,
                    linear=energy,
                    db=amplitude_to_db(math.sqrt(energy)),
                    density_linear=density,
                    density_db=amplitude_to_db(math.sqrt(density)),
                )
            )

        features = FrameFeatures(
            index=self._index,
            start_sample=int(start_sample),
            sample_rate=self.sample_rate,
            hop_samples=0,
            rms=rms,
            peak=peak,
            crest_factor=crest,
            zero_crossing_rate=zcr,
            dc_offset=dc_offset,
            rms_db=amplitude_to_db(rms),
            peak_db=amplitude_to_db(peak),
            spectral_centroid_hz=centroid,
            spectral_bandwidth_hz=bandwidth,
            spectral_rolloff_hz=rolloff,
            spectral_flatness=flatness,
            spectral_flux=flux,
            dominant_frequency_hz=dominant,
            spectral_peakiness=peakiness,
            bands=tuple(band_values),
        )
        self._index += 1
        return features

    # ------------------------------------------------------------------
    def analyse_stream(
        self, samples: np.ndarray, hop_samples: int, start_sample: int = 0
    ) -> list[FrameFeatures]:
        """Analyse a whole array, one frame per hop.  Convenience for tests/CLI."""
        from .frames import FrameExtractor

        extractor = FrameExtractor(
            samples.size if samples.size else 1, hop_samples, None
        )  # raw frames; the extractor windows internally
        frames = extractor.push(np.asarray(samples, dtype=np.float32))
        out: list[FrameFeatures] = []
        for index, frame in enumerate(frames):
            out.append(self.analyse(frame, start_sample + index * hop_samples))
        return out


def band_labels() -> tuple[str, ...]:
    return BAND_LABELS
