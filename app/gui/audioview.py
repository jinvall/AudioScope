"""Waveform and spectrogram computation for the review GUI.

Deliberately free of any Qt import, so the maths is testable headless and
reusable. The widgets in :mod:`app.gui.widgets` only turn these results into
``QImage``/``QPainter`` calls.

Two rules from the brief drive the design:

* **Lazy.** A waveform is produced when an event is *selected*, never for the
  whole list. A spectrogram is produced only when the user asks to see one.
* **Reuse the backend.** The spectrogram runs the same
  :class:`~app.analysis.features.FeatureExtractor` and framing the live pipeline
  uses, rather than introducing a second analysis path for display. The
  waveform is a min/max envelope, which is cheap and shows transients that a
  mean would hide.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

#: Columns produced for a full-width waveform.  One min/max pair per column
#: keeps the cost independent of how long the event is.
DEFAULT_COLUMNS = 1200

#: Cap on stored envelope resolution, so a very long event cannot produce an
#: unbounded array.
MAX_COLUMNS = 4000


class AudioUnavailable(Exception):
    """Raised when an event's audio cannot be read.

    Distinct from a decoding error so the GUI can say "no audio" rather than
    "corrupt audio", and so the event remains fully usable either way.
    """


class AudioCorrupt(AudioUnavailable):
    """Raised when the audio exists but cannot be decoded."""


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def read_event_audio(path: Optional[str], target_rate: int) -> np.ndarray:
    """Read an event's audio as mono float32 at ``target_rate``.

    Raises :class:`AudioUnavailable` when the file is missing and
    :class:`AudioCorrupt` when it exists but cannot be decoded. The GUI shows
    metadata and keeps annotation available in both cases, so these have to be
    distinguishable.
    """
    if not path:
        raise AudioUnavailable("no audio recorded for this event")
    if not os.path.exists(path):
        raise AudioUnavailable(f"audio file not found: {path}")
    try:
        from ..audio.wavio import read_wav_mono, probe_wav

        info = probe_wav(path)
        samples = read_wav_mono(path)
    except FileNotFoundError as exc:
        raise AudioUnavailable(f"audio file not found: {path}") from exc
    except Exception as exc:
        raise AudioCorrupt(f"cannot read audio: {exc}") from exc

    if samples.size == 0:
        raise AudioCorrupt("audio file contains no samples")
    if info.sample_rate != target_rate:
        # Reuse the streaming resampler the capture path already uses rather
        # than adding a display-only resampler.
        from ..audio.resample import StreamingResampler, ratio_for

        up, down = ratio_for(info.sample_rate, target_rate)
        samples = StreamingResampler(up, down).process(samples)
    return np.ascontiguousarray(samples, dtype=np.float32)


# ----------------------------------------------------------------------
# Waveform
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Envelope:
    """A min/max envelope, the honest way to draw a waveform.

    A mean would render an AC-coupled or low-amplitude signal as a flat line and
    hide transients, which are exactly what a reviewer is looking for.
    """

    peaks: np.ndarray
    troughs: np.ndarray
    sample_rate: int
    samples: int
    peak: float

    @property
    def duration(self) -> float:
        return self.samples / self.sample_rate if self.sample_rate else 0.0

    @property
    def columns(self) -> int:
        return int(self.peaks.size)

    def bounds_at(self, fraction: float) -> tuple:
        """Peak and trough for a column, for a click or hover read-out."""
        index = int(max(0.0, min(1.0, fraction)) * (self.columns - 1))
        index = max(0, min(self.columns - 1, index))
        return float(self.troughs[index]), float(self.peaks[index])


def build_envelope(
    samples: np.ndarray,
    sample_rate: int,
    columns: int = DEFAULT_COLUMNS,
) -> Envelope:
    """Reduce samples to a min/max envelope of at most ``columns`` points."""
    data = np.asarray(samples, dtype=np.float32).ravel()
    if data.size == 0:
        empty = np.zeros(0, dtype=np.float32)
        return Envelope(empty, empty, sample_rate, 0, 0.0)
    count = int(max(1, min(columns, MAX_COLUMNS, data.size)))
    # np.array_split avoids an index array and keeps the cost O(n).
    chunks = np.array_split(data, count)
    peaks = np.array(
        [float(c.max()) if c.size else 0.0 for c in chunks], dtype=np.float32
    )
    troughs = np.array(
        [float(c.min()) if c.size else 0.0 for c in chunks], dtype=np.float32
    )
    return Envelope(
        peaks=peaks,
        troughs=troughs,
        sample_rate=sample_rate,
        samples=int(data.size),
        peak=float(max(abs(peaks.max()), abs(troughs.min()))) if count else 0.0,
    )


# ----------------------------------------------------------------------
# Spectrogram
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Spectrogram:
    """A dB-scaled spectrogram for one event.

    ``data`` is ``(bins, columns)`` of intensities in 0..1, ready for a colour
    lookup. Produced by the same feature extractor the live pipeline uses, so
    the GUI and the detector see the same analysis.
    """

    data: np.ndarray
    frequencies: np.ndarray
    sample_rate: int
    max_hz: float

    @property
    def bins(self) -> int:
        return int(self.data.shape[0]) if self.data.size else 0

    @property
    def columns(self) -> int:
        return int(self.data.shape[1]) if self.data.size else 0


def build_spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    config=None,
    columns: int = 512,
    max_hz: float = 12000.0,
) -> Spectrogram:
    """Build an event-scoped spectrogram using the backend's own analysis.

    Reuses :class:`~app.analysis.features.FeatureExtractor` and the framing in
    :class:`~app.analysis.frames.FrameExtractor` with the configured analysis
    window, so this is the same measurement the detector makes - not a second,
    display-only pipeline.
    """
    from ..analysis.features import FeatureExtractor
    from ..analysis.frames import FrameExtractor
    from ..config import AppConfig

    config = config or AppConfig().validate()
    analysis = config.analysis
    frame = analysis.frame_samples_at(sample_rate)
    hop = max(1, int(round(analysis.hop_ms * sample_rate / 1000.0)))

    extractor = FeatureExtractor(
        sample_rate,
        rolloff_percentile=analysis.rolloff_percentile,
        window=analysis.window,
    )
    framer = FrameExtractor(frame, hop, None)

    data = np.asarray(samples, dtype=np.float32).ravel()
    if data.size == 0:
        return Spectrogram(
            np.zeros((0, 0), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            sample_rate,
            max_hz,
        )

    # Frames are stacked down to the requested column count: bounded work, and
    # independent of event length.
    wanted = int(max(8, min(columns, data.size // max(1, hop) or 1)))
    stride = max(1, len(range(0, max(1, data.size - frame), hop)) // wanted)

    columns_out = []
    for index, chunk in enumerate(framer.push(data)):
        if index % stride:
            continue
        features = extractor.analyse(chunk)
        spectrum = extractor.last_magnitude
        if spectrum is None:
            continue
        columns_out.append(np.asarray(spectrum, dtype=np.float32))
        if len(columns_out) >= wanted:
            break

    if not columns_out:
        return Spectrogram(
            np.zeros((0, 0), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            sample_rate,
            max_hz,
        )

    matrix = np.stack(columns_out, axis=1)          # (bins, columns)
    frequencies = extractor.frequencies
    keep = frequencies <= min(max_hz, sample_rate / 2.0)
    matrix = matrix[keep]
    frequencies = frequencies[keep]

    # Per-column normalisation, matching the live SpectrogramBuffer, so a quiet
    # event still shows structure instead of a black rectangle.
    peak = float(matrix.max()) if matrix.size else 0.0
    if peak > 0:
        db = 20.0 * np.log10(np.maximum(matrix, 1e-10) / peak)
        floor_db = -80.0
        db = np.clip(db, floor_db, 0.0)
        image = ((db - floor_db) / -floor_db).astype(np.float32)
    else:
        image = np.zeros_like(matrix, dtype=np.float32)

    return Spectrogram(
        data=image, frequencies=frequencies, sample_rate=sample_rate,
        max_hz=float(frequencies[-1]) if frequencies.size else max_hz,
    )


def ramp_lut(colors, steps: int = 256) -> np.ndarray:
    """Build a 256-entry RGB lookup table from the theme's spectral stops.

    The theme pack supplies the spectral gradient, so the spectrogram uses the
    intended colour ramp rather than an arbitrary colormap. Values below the
    first stop take the first stop's colour.
    """
    palette = []
    for color in colors:
        text = str(color).lstrip("#")
        if len(text) >= 6:
            try:
                palette.append(
                    (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
                )
            except ValueError:
                continue
    if not palette:
        palette = [(0, 0, 0), (255, 255, 255)]
    lut = np.zeros((steps, 3), dtype=np.uint8)
    n = len(palette)
    for index in range(steps):
        position = (index / max(steps - 1, 1)) * (n - 1)
        low = int(math.floor(position))
        high = min(low + 1, n - 1)
        weight = position - low
        for channel in range(3):
            lut[index, channel] = int(round(
                palette[low][channel] * (1 - weight)
                + palette[high][channel] * weight
            ))
    return lut
