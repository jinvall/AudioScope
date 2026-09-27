"""Thread-safe rolling audio buffer.

docs/AUDIO_PIPELINE.md section 3.  The buffer exists so that the event system
can extract audio that occurred *before* the detector decided an event
happened (pre-roll, AGENTS.md section 5).  That is only possible if the
buffer is indexed by an absolute sample timeline that keeps increasing, rather
than by position within the current contents.

Design
------
A preallocated float32 array of ``capacity`` frames plus:

* ``_write``  - index of the next write position, modulo capacity
* ``_total``  - frames ever written; never decreases, so it *is* the absolute
  sample clock shared with the rest of the pipeline

Every read takes an absolute frame range.  Reads that fall outside the
retained window are clamped, and :attr:`shortfall` reports how much requested
history was not available, so callers never silently receive less audio than
they asked for.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np


class RingBuffer:
    """Fixed-capacity mono float32 ring buffer.

    Parameters
    ----------
    sample_rate:
        Frame rate, used to convert between frames and seconds.
    seconds:
        Retained duration.  Callers should validate this against
        :data:`app.config.RING_BUFFER_MIN_SECONDS` /
        :data:`app.config.RING_BUFFER_MAX_SECONDS`.
    """

    def __init__(self, sample_rate: int, seconds: float) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if seconds <= 0:
            raise ValueError("seconds must be positive")
        self.sample_rate = int(sample_rate)
        self.seconds = float(seconds)
        self.capacity = int(round(self.seconds * self.sample_rate))
        if self.capacity <= 0:
            raise ValueError(
                f"{seconds}s at {sample_rate} Hz is not a whole frame count"
            )
        self._data = np.zeros(self.capacity, dtype=np.float32)
        self._write = 0
        self._total = 0
        self._lock = threading.Lock()
        # Frames lost because a writer overwrote them faster than a reader
        # consumed them.  Non-zero means the pipeline is too slow.
        self._overruns = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def total_written(self) -> int:
        """Frames ever written; the buffer's absolute sample clock."""
        with self._lock:
            return self._total

    @property
    def frames_available(self) -> int:
        """Frames currently retained."""
        with self._lock:
            return min(self._total, self.capacity)

    @property
    def available_seconds(self) -> float:
        return self.frames_available / self.sample_rate

    @property
    def overruns(self) -> int:
        """Frames overwritten before any reader could have consumed them."""
        with self._lock:
            return self._overruns

    @property
    def shortfall(self) -> int:
        """Frames of history lost to capacity (0 until 2x capacity seen)."""
        with self._lock:
            return max(0, self._total - self.capacity)

    @property
    def oldest_available(self) -> int:
        """Absolute frame index of the oldest retained frame."""
        with self._lock:
            return max(0, self._total - self.capacity)

    @property
    def newest_available(self) -> int:
        """Absolute frame index one past the newest retained frame."""
        with self._lock:
            return self._total

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def write(self, samples: np.ndarray) -> int:
        """Append mono samples.  Returns the number of frames written.

        Never blocks and never allocates beyond the preallocated array: the
        capture callback calls this, so it must stay cheap (AGENTS.md 2.3).
        Samples that would fall outside the capacity are counted in
        :attr:`overruns` rather than being silently lost.
        """
        block = np.asarray(samples, dtype=np.float32).ravel()
        n = block.size
        if n == 0:
            return 0

        with self._lock:
            # The sample clock always advances by the full amount, even when
            # only the tail can be retained.  Absolute addressing across the
            # whole session depends on that: a reader asking for a range must
            # learn that the data is gone (via `shortfall`) rather than being
            # handed a silently shifted window.
            dropped = max(0, n - self.capacity)
            if dropped:
                block = block[dropped:]
                # Frames that arrived with nowhere to go.  Counted so the
                # pipeline can report a real loss of history rather than
                # quietly serving a shifted window.
                self._overruns += dropped
            stored = n - dropped

            # Place the block at the array slot its *true* global index maps
            # to.  When data is dropped this is not `_write`, and using
            # `_write` here would shift every retained frame by `dropped`.
            start_global = self._total + dropped
            pos = start_global % self.capacity
            end = pos + stored
            if end <= self.capacity:
                self._data[pos:end] = block
            else:
                first = self.capacity - pos
                self._data[pos:] = block[:first]
                self._data[:stored - first] = block[first:]

            self._write = (start_global + stored) % self.capacity
            self._total += n
        return n

    def clear(self) -> None:
        """Drop all history and reset the sample clock."""
        with self._lock:
            self._data[:] = 0.0
            self._write = 0
            self._total = 0
            self._overruns = 0

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def read(self, start: int, end: int) -> np.ndarray:
        """Read the absolute frame range ``[start, end)``.

        The range is clamped to what is actually retained.  Use
        :meth:`read_exact` when the caller must know whether it got everything.
        """
        if end <= start:
            return np.zeros(0, dtype=np.float32)
        with self._lock:
            lo = max(int(start), self._total - self.capacity, 0)
            hi = min(int(end), self._total)
            if hi <= lo:
                return np.zeros(0, dtype=np.float32)
            n = hi - lo
            out = np.empty(n, dtype=np.float32)
            first = min(n, self.capacity - (lo % self.capacity))
            out[:first] = self._data[(lo % self.capacity):(lo % self.capacity) + first]
            if n > first:
                out[first:] = self._data[:n - first]
            return out

    def read_exact(self, start: int, end: int) -> tuple[np.ndarray, int]:
        """Read ``[start, end)`` and report how many frames were unavailable.

        Returns ``(samples, missing)`` where ``missing`` counts requested
        frames that had already been overwritten.  ``missing == 0`` means the
        full range was genuinely in the buffer.
        """
        with self._lock:
            oldest = max(0, self._total - self.capacity)
            start = int(start)
            # The retained window is [oldest, total).  A request can be short in
            # two distinct ways and both are real losses:
            #
            #  * before the retained window, because the data was overwritten;
            #  * beyond everything ever written, because the stream has not been
            #    captured that far yet.
            #
            # The second case is the one that matters for a pre-roll early in a
            # session: the buffer is mostly empty, so `oldest` is 0 and without
            # this the missing history would be reported as zero even though
            # none of it was ever captured.
            missing = max(0, oldest - start) + max(0, start - self._total)
        return self.read(start, end), missing

    def read_latest(self, frames: int) -> np.ndarray:
        """Read the most recent ``frames`` frames."""
        with self._lock:
            end = self._total
        return self.read(end - frames, end)

    def read_seconds_latest(self, seconds: float) -> np.ndarray:
        frames = int(round(seconds * self.sample_rate))
        return self.read_latest(frames)

    def pre_roll(self, at_frame: int, frames: int) -> tuple[np.ndarray, int]:
        """Extract the ``frames`` immediately *before* ``at_frame``.

        This is the pre-roll extraction required by docs/AUDIO_PIPELINE.md
        section 9.  Returns ``(samples, missing)``.
        """
        start = at_frame - frames
        return self.read_exact(start, at_frame)

    def snapshot(self) -> np.ndarray:
        """Copy the entire retained history, oldest first."""
        with self._lock:
            end = self._total
            start = max(0, end - self.capacity)
        return self.read(start, end)

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------
    def frame_to_seconds(self, frame: int) -> float:
        return frame / self.sample_rate

    def seconds_to_frame(self, seconds: float) -> int:
        return int(round(seconds * self.sample_rate))

    def time_of_latest_sample(self) -> float:
        """Seconds of stream time represented by the newest frame."""
        return self.total_written / self.sample_rate

    def seconds_until(self, frame: int) -> Optional[float]:
        """Seconds of retained history before ``frame``, or None if past."""
        with self._lock:
            oldest = max(0, self._total - self.capacity)
        if frame < oldest:
            return None
        return (frame - oldest) / self.sample_rate

    def __len__(self) -> int:
        return self.capacity

    def __repr__(self) -> str:
        return (
            f"<RingBuffer {self.seconds:g}s @ {self.sample_rate} Hz, "
            f"capacity={self.capacity}, held={self.frames_available}>"
        )
