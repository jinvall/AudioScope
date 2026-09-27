"""Frame extraction for continuous analysis.

docs/AUDIO_PIPELINE.md section 4: the analysis worker operates on short
overlapping frames, and detector timing must not depend on an arbitrary frame
size.

Two consequences shape this module:

* Frames are cut on a fixed **hop grid** measured from the start of the stream,
  not from wherever a block happens to end.  Feeding 1024-sample blocks or
  777-sample blocks therefore yields frames at identical positions, so feature
  values do not shift when the device's block size changes.
* The consumer may be fed blocks of any size and cadence.  Samples are buffered
  until a whole frame is available and the remainder is carried forward, so no
  audio is lost at a block boundary.

Not thread-safe: this belongs to the analysis worker thread only.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Analysis band edges in Hz (docs/AUDIO_PIPELINE.md section 5, AGENTS.md 7).
# The top band stops at 24 kHz rather than at Nyquist so the set is independent
# of the sample rate and stays comparable between runs at different rates.
ANALYSIS_BANDS: tuple[tuple[float, float], ...] = (
    (20.0, 80.0),
    (80.0, 250.0),
    (250.0, 500.0),
    (500.0, 1000.0),
    (1000.0, 2000.0),
    (2000.0, 4000.0),
    (4000.0, 8000.0),
    (8000.0, 16000.0),
    (16000.0, 24000.0),
)

BAND_LABELS: tuple[str, ...] = tuple(f"{lo:g}-{hi:g}" for lo, hi in ANALYSIS_BANDS)


def window_function(name: str, length: int) -> np.ndarray:
    """Return a window of the requested type.

    ``"none"`` and ``"rect"`` both mean no windowing.  A rectangular window is
    correct for measuring level but leaks badly for anything spectral, so it is
    never the default.
    """
    n = int(length)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.ones(1, dtype=np.float64)
    if name in ("none", "rect"):
        return np.ones(n, dtype=np.float64)
    if name == "hann":
        return np.hanning(n)
    if name == "hamming":
        return np.hamming(n)
    if name == "blackman":
        return np.blackman(n)
    raise ValueError(f"unknown window {name!r}")


class FrameExtractor:
    """Cuts a sample stream into fixed-length frames on a fixed hop grid.

    Parameters
    ----------
    frame_samples:
        Samples per frame.
    hop_samples:
        Samples between frame starts.  Must be ``1 <= hop <= frame``.
    window:
        Window name applied to each emitted frame, or ``None`` to leave the
        frames unwindowed.  Level features should use no window; spectral
        features should use one.
    """

    def __init__(
        self,
        frame_samples: int,
        hop_samples: int,
        window: Optional[str] = None,
    ) -> None:
        if frame_samples <= 0:
            raise ValueError("frame_samples must be positive")
        if hop_samples <= 0:
            raise ValueError("hop_samples must be positive")
        if hop_samples > frame_samples:
            raise ValueError(
                f"hop_samples ({hop_samples}) must not exceed frame_samples "
                f"({frame_samples}); a hop longer than the frame would skip audio"
            )
        self.frame_samples = int(frame_samples)
        self.hop_samples = int(hop_samples)
        self.window_name = window
        self._window = (
            np.ones(self.frame_samples, dtype=np.float32)
            if window in (None, "none", "rect")
            else window_function(window, self.frame_samples).astype(np.float32)
        )

        self._buf = np.zeros(0, dtype=np.float32)
        # Absolute stream index of _buf[0].
        self._buf_start = 0
        # Absolute index of the next frame to emit.
        self._next_start = 0
        self._total_in = 0

        self.frames_emitted = 0
        self.samples_in = 0
        self._padded_frames = 0
        self._tail_unanalysed = 0

    # ------------------------------------------------------------------
    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        """Add samples and return every frame that is now complete."""
        block = np.asarray(samples, dtype=np.float32).ravel()
        if block.size == 0:
            return []

        if self._buf.size:
            self._buf = np.concatenate((self._buf, block))
        else:
            self._buf = block.copy()
        self._total_in += int(block.size)
        self.samples_in += int(block.size)

        frames: list[np.ndarray] = []
        limit = self._buf_start + self._buf.size
        while self._next_start + self.frame_samples <= limit:
            local = self._next_start - self._buf_start
            frames.append(
                self._buf[local:local + self.frame_samples] * self._window
            )
            self._next_start += self.hop_samples
            self.frames_emitted += 1

        # Drop everything before the next frame start: no future frame needs it.
        # Without this the buffer would grow for the life of the session
        # (docs/PERFORMANCE.md section 7).
        if self._next_start > self._buf_start:
            drop = min(self._next_start - self._buf_start, self._buf.size)
            if drop > 0:
                self._buf = self._buf[drop:]
                self._buf_start += drop
        return frames

    # ------------------------------------------------------------------
    def flush(self, pad: bool = False) -> list[np.ndarray]:
        """Handle the tail of the stream at end of input.

        By default the incomplete tail is **discarded and counted** in
        :attr:`tail_samples_unanalysed`.  Padding it with zeros would be
        actively harmful: a padded frame is mostly digital silence, and
        feeding that to an adaptive noise floor teaches it that the room is
        silent.  A monitor must not invent audio it did not capture.

        Pass ``pad=True`` to emit one zero-padded frame anyway, for callers
        that need a fixed frame count (a test fixture, say).  Such frames are
        counted in :attr:`padded_frames` so a consumer can discount them.
        """
        tail = max(0, self._buf.size - (self._next_start - self._buf_start))
        if tail <= 0:
            return []
        if not pad:
            self._tail_unanalysed = tail
            self._buf = np.zeros(0, dtype=np.float32)
            self._buf_start = self._next_start
            return []

        local = self._next_start - self._buf_start
        frame = np.zeros(self.frame_samples, dtype=np.float32)
        available = min(self.frame_samples, self._buf.size - local)
        # The window must be sliced to match: the tail of the frame is padding.
        frame[:available] = (
            self._buf[local:local + available] * self._window[:available]
        )
        self._next_start += self.hop_samples
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = self._next_start
        self.frames_emitted += 1
        self._padded_frames += 1
        return [frame]

    @property
    def padded_frames(self) -> int:
        """Frames emitted only because ``pad=True`` was requested."""
        return self._padded_frames

    @property
    def tail_samples_unanalysed(self) -> int:
        """Samples at the end of the stream too short to form a frame.

        Bounded by ``frame_samples`` and normally zero.  Reported so a caller
        can say how much of the input was not analysed.
        """
        return self._tail_unanalysed

    @property
    def buffered_samples(self) -> int:
        return int(self._buf.size)

    @property
    def position(self) -> int:
        """Absolute index of the next frame to be emitted."""
        return int(self._next_start)

    def reset(self) -> None:
        """Discard buffered audio and restart the hop grid."""
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = 0
        self._next_start = 0
        self._total_in = 0
        self.frames_emitted = 0
        self.samples_in = 0
        self._padded_frames = 0
        self._tail_unanalysed = 0
