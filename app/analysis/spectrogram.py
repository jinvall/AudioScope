"""Rolling spectrogram and waveform buffers for display.

docs/AUDIO_PIPELINE.md section 11: the spectrogram uses the same event time
base as the waveform, so event markers align across both views.  Both buffers
here share the analysis hop, so a column index means the same instant in the
waveform, the spectrogram and the event timeline.

These are display structures, not analysis.  They hold dB-scaled magnitudes and
peak envelopes; they never touch the stored evidence audio, which is only ever
read from the lossless recording or the ring buffer.

Thread-safe: the analysis worker writes, the GUI reads.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

# Lowest level a spectrogram column may show, in dB below the frame maximum.
# Anything quieter is at the bottom of the colour range.  Keeping the scale
# relative means a quiet room still shows structure instead of a black image.
DEFAULT_FLOOR_DB = -80.0


class SpectrogramBuffer:
    """Fixed-size ring of dB-scaled magnitude columns.

    Parameters
    ----------
    bins:
        FFT bins per column (rfft output length).
    columns:
        How many time columns to keep.
    max_hz:
        Highest frequency shown.  Bins above it are dropped, so a 48 kHz stream
        and a 44.1 kHz stream can be compared on the same axis.
    floor_db:
        Column floor relative to that column's own peak.
    """

    def __init__(
        self,
        bins: int,
        columns: int,
        max_hz: float = 24000.0,
        sample_rate: int = 48000,
        floor_db: float = DEFAULT_FLOOR_DB,
    ) -> None:
        if bins <= 0:
            raise ValueError("bins must be positive")
        if columns <= 0:
            raise ValueError("columns must be positive")
        self.bins = int(bins)
        self.requested_columns = int(columns)
        self.sample_rate = int(sample_rate)
        self.floor_db = float(floor_db)

        self.max_hz = float(min(max_hz, self.sample_rate / 2.0))
        full = np.fft.rfftfreq(self.bins * 2, d=1.0 / self.sample_rate)
        self._keep = full <= self.max_hz
        if not np.any(self._keep):
            self._keep = np.ones_like(full, dtype=bool)
        self.kept_bins = int(np.count_nonzero(self._keep))
        self.frequencies = full[self._keep]

        # Columns are stored newest-last; the newest is always at index -1.
        self._data = np.zeros((self.requested_columns, self.kept_bins),
                              dtype=np.float32)
        self._write = 0
        self._filled = 0
        self._lock = threading.Lock()
        self.frames_pushed = 0

    # ------------------------------------------------------------------
    def push(self, magnitude: np.ndarray) -> None:
        """Add one magnitude spectrum as the newest column."""
        data = np.asarray(magnitude, dtype=np.float32).ravel()
        if data.size < self.kept_bins:
            padded = np.zeros(self.kept_bins, dtype=np.float32)
            padded[: data.size] = data
            data = padded
        elif data.size > self.kept_bins:
            data = data[: self.kept_bins]

        peak = float(np.max(data))
        if peak > 0.0:
            db = 20.0 * np.log10(np.maximum(data, 1e-10) / peak)
        else:
            db = np.full(self.kept_bins, self.floor_db, dtype=np.float32)
        np.clip(db, self.floor_db, 0.0, out=db)
        # Map to 0..1 for direct use as a colour intensity.
        column = ((db - self.floor_db) / -self.floor_db).astype(np.float32)

        with self._lock:
            self._data[self._write] = column
            self._write = (self._write + 1) % self.requested_columns
            self._filled = min(self._filled + 1, self.requested_columns)
        self.frames_pushed += 1

    # ------------------------------------------------------------------
    @property
    def filled_columns(self) -> int:
        with self._lock:
            return self._filled

    @property
    def is_full(self) -> bool:
        return self.filled_columns >= self.requested_columns

    def image(self) -> np.ndarray:
        """The buffer as a ``(columns, bins)`` array, oldest first.

        Rows are normalised intensities in 0..1.  Unfilled leading rows are
        zeros, so the image lines up with :attr:`age_seconds` and the event
        timeline without any further bookkeeping by the caller.
        """
        with self._lock:
            if self._filled < self.requested_columns:
                out = np.zeros_like(self._data)
                out[self.requested_columns - self._filled:] = self._data[:self._filled]
                return out
            # Full: oldest sits just after the write cursor.
            head = self._data[self._write:]
            tail = self._data[:self._write]
            return np.concatenate((head, tail), axis=0)

    def column_at(self, age_columns: int = 0) -> np.ndarray:
        """One column, ``0`` being the newest.  Out of range gives zeros."""
        if age_columns < 0 or age_columns >= self.requested_columns:
            return np.zeros(self.kept_bins, dtype=np.float32)
        with self._lock:
            index = (self._write - 1 - age_columns) % self.requested_columns
            return self._data[index].copy()

    def clear(self) -> None:
        with self._lock:
            self._data[:] = 0.0
            self._write = 0
            self._filled = 0
        self.frames_pushed = 0

    # ------------------------------------------------------------------
    @property
    def column_seconds(self) -> float:
        """Seconds per column; set by the owner to the analysis hop."""
        return getattr(self, "_column_seconds", 0.0)

    @column_seconds.setter
    def column_seconds(self, value: float) -> None:
        if value <= 0:
            raise ValueError("column_seconds must be positive")
        self._column_seconds = float(value)

    @property
    def span_seconds(self) -> float:
        return self.requested_columns * self.column_seconds

    @property
    def age_seconds(self) -> float:
        """Seconds between the newest column and the left edge of the image."""
        return (self.requested_columns - self.filled_columns) * self.column_seconds


class WaveformBuffer:
    """Rolling min/max envelope for the live waveform display.

    Each column holds a peak pair, so a waveform drawn from this shows the same
    extent as the audio rather than a mean that hides transients - a mean makes
    an AC-coupled signal look flat.
    """

    def __init__(self, columns: int = 1024) -> None:
        if columns <= 0:
            raise ValueError("columns must be positive")
        self.columns = int(columns)
        self._peaks = np.zeros(self.columns, dtype=np.float32)
        self._troughs = np.zeros(self.columns, dtype=np.float32)
        self._write = 0
        self._filled = 0
        self._lock = threading.Lock()
        self.samples_pushed = 0

    def push(self, samples: np.ndarray) -> None:
        """Add samples, downsampled to one min/max column."""
        data = np.asarray(samples, dtype=np.float32).ravel()
        if data.size == 0:
            return
        # One column per group of samples; simplest honest choice is one
        # column per call, with the group's extremes recorded.
        with self._lock:
            self._peaks[self._write] = float(np.max(data))
            self._troughs[self._write] = float(np.min(data))
            self._write = (self._write + 1) % self.columns
            self._filled = min(self._filled + 1, self.columns)
        self.samples_pushed += int(data.size)

    def envelope(self) -> tuple[np.ndarray, np.ndarray]:
        """``(peaks, troughs)`` oldest first."""
        with self._lock:
            if self._filled < self.columns:
                peaks = np.zeros(self.columns, dtype=np.float32)
                troughs = np.zeros(self.columns, dtype=np.float32)
                peaks[self.columns - self._filled:] = self._peaks[:self._filled]
                troughs[self.columns - self._filled:] = self._troughs[:self._filled]
                return peaks, troughs
            head = self._peaks[self._write:]
            tail = self._peaks[:self._write]
            peaks = np.concatenate((head, tail))
            head = self._troughs[self._write:]
            tail = self._troughs[:self._write]
            troughs = np.concatenate((head, tail))
        return peaks, troughs

    @property
    def filled_columns(self) -> int:
        with self._lock:
            return self._filled

    def clear(self) -> None:
        with self._lock:
            self._peaks[:] = 0.0
            self._troughs[:] = 0.0
            self._write = 0
            self._filled = 0
        self.samples_pushed = 0


def write_pgm(
    path: str,
    image: np.ndarray,
    axis: Optional[np.ndarray] = None,
    renormalise: bool = True,
) -> str:
    """Write a spectrogram image as a binary PGM, with no dependencies.

    ``image`` is ``(columns, bins)`` with values nominally in 0..1.  Columns run
    oldest to newest, so the left edge of the file is the oldest audio.

    The live buffer normalises each column to its own peak, which is right for a
    scrolling view - the spectrum shape is always visible, however quiet the
    moment was.  That same normalisation makes a *static* image flat, because
    every column reaches full brightness and the time structure disappears.  So
    by default the export rescales against the global maximum, which shows the
    dynamics across the whole span.  Pass ``renormalise=False`` to keep the
    per-column scaling.

    PGM exists so the spectrogram can be inspected on a machine with nothing
    but the standard library and numpy.  A PNG is nicer, but it needs
    matplotlib, and a display feature should not become a hard dependency.

    When ``axis`` is given, the frequency range is appended as a PGM comment so
    the image can be interpreted without the original code.
    """
    data = np.asarray(image, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("image must be 2-D (columns, bins)")
    scaled = np.clip(data, 0.0, 1.0)
    if renormalise:
        peak = float(scaled.max())
        if peak > 0.0:
            scaled = scaled / peak
    pixels = np.rint(scaled * 255.0).astype(np.uint8)
    height, width = pixels.shape  # rows = bins (frequency), cols = time

    header = [f"P5\n# Audio Microscope spectrogram".encode("ascii")]
    if axis is not None and np.asarray(axis).size == height:
        axis = np.asarray(axis, dtype=np.float64)
        header.append(
            f"# frequency 0..{axis[-1]:.1f} Hz over {height} rows".encode("ascii")
        )
    header.append(f"# time 0..{width} columns, oldest first".encode("ascii"))
    header.append(f"{width} {height}\n255\n".encode("ascii"))

    with open(path, "wb") as handle:
        for part in header:
            handle.write(part)
        # PGM rows run top to bottom; put the highest frequency at the top so
        # the image matches how a spectrogram is normally read.
        handle.write(np.flipud(pixels).tobytes())
    return path
