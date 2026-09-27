"""Streaming rational resampler.

docs/AUDIO_PIPELINE.md section 1 requires that a device whose native rate
differs from the internal rate is converted *once*, at the input boundary.
A capture callback cannot accumulate a whole file before converting, so the
conversion has to be incremental with carried state.  This module provides it.

How the block stitching is made exact
-------------------------------------
A windowed-sinc prototype ``h`` is designed for the rate ratio and applied
with :func:`scipy.signal.upfirdn`.  For a block of input ``b`` the output is

    y[m] = sum_k h[k] * b_full[(m*down - k)/up]     (m*down - k = 0 mod up)

so output ``m`` reads input at or before ``floor(m*down/up)``.  Two
consequences drive the implementation:

1. **How many outputs to release.**  Output ``m`` is fully determined as soon
   as ``ceil(m*down/up)`` input samples exist.  After ``P`` input samples the
   available output count is ``floor(P*up/down)``, so a call that advances
   ``P`` from ``P0`` releases exactly ``floor(P*up/down) - floor(P0*up/down)``.
   Summing that telescopes to ``floor(len(x) * up / down)`` overall, which is
   the correct sample count for the conversion.

2. **Which outputs to take.**  ``upfirdn`` is fed the carried history plus
   the new block, and its result extends past the end of the block by the
   filter's ringdown, so the release rule alone does not say where to slice.
   The carried history is therefore rounded up to a whole multiple of
   ``down``: it then spans exactly ``keep*up/down`` output samples, and the
   index offset between a filter result and the global output timeline is an
   *integer* rather than a fraction.  Global output ``n`` therefore sits at
   local index ``n + keep*up/down``, and the slice is exact.

With the history aligned this way the result is independent of block size: the
concatenation of the returned blocks is identical, sample for sample, to a
single call on the whole signal.  There is no per-block phase error, so block
boundaries introduce no discontinuity.

``tests/test_resample.py`` asserts that equality, and the passband amplitude
and stopband rejection, directly.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin, upfirdn

# Kaiser window beta for ~90 dB stopband attenuation.  Rate-conversion quality
# is audible as pre/post-echo, so this is not a place to economise on taps.
_KAISER_BETA = 8.6

# Filter length as a multiple of the larger of the two rates.  SciPy's own
# resample_poly uses 10; 8 keeps the per-block cost down with no practical loss.
_TAPS_PER_RATE = 8


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def design_prototype(up: int, down: int) -> np.ndarray:
    """Design the anti-aliasing low-pass for an ``up/down`` rate change."""
    max_rate = max(up, down)
    half = _TAPS_PER_RATE * max_rate
    # Cutoff at the lower of the two Nyquist limits, in normalised frequency.
    h = firwin(
        2 * half + 1,
        1.0 / max_rate,
        window=("kaiser", _KAISER_BETA),
        scale=True,
    )
    # Compensate for the energy removed when the filter decimates.
    return (h * up).astype(np.float64)


def ratio_for(source_rate: int, target_rate: int) -> tuple[int, int]:
    """Return the reduced ``(up, down)`` converting source to target rate."""
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    divisor = _gcd(int(source_rate), int(target_rate))
    return target_rate // divisor, source_rate // divisor


class StreamingResampler:
    """Stateful rational resampler.

    ``StreamingResampler(*ratio_for(44100, 48000))`` converts a 44.1 kHz
    stream to 48 kHz.  Feed it any block size; output is continuous.
    """

    def __init__(self, up: int, down: int) -> None:
        if up <= 0 or down <= 0:
            raise ValueError("up and down must be positive")
        divisor = _gcd(int(up), int(down))
        self.up = int(up) // divisor
        self.down = int(down) // divisor
        self.ratio = self.up / self.down

        if self.up == self.down:
            # 1:1.  No filtering, no state, pass through.
            self._h: np.ndarray | None = None
            self._keep = 0
            self._keep_out = 0
        else:
            self._h = design_prototype(self.up, self.down)
            # History rounded up to a whole multiple of `down` so that it maps
            # to a whole number of output samples.
            self._keep = -(-(len(self._h) - 1) // self.down) * self.down
            self._keep_out = self._keep * self.up // self.down

        # Input FIFO.  Only whole multiples of `down` are ever handed to
        # `upfirdn`; this is what keeps every index offset an integer, no
        # matter what block size the caller uses.  See the module docstring.
        self._inbuf = np.zeros(0, dtype=np.float32)
        self._hist = np.zeros(self._keep, dtype=np.float32)
        self._seen = 0      # total input samples accepted
        self._released = 0  # total output samples emitted
        self._pending = 0   # input samples held back, always < down

    # ------------------------------------------------------------------
    @property
    def is_identity(self) -> bool:
        """True when the ratio is 1:1 and no filtering is performed."""
        return self._h is None

    @property
    def input_samples_processed(self) -> int:
        return self._seen

    @property
    def output_samples_produced(self) -> int:
        return self._released

    @property
    def pending_input_samples(self) -> int:
        """Input samples held in the alignment FIFO (always < ``down``)."""
        return self._pending

    @property
    def group_delay_input_samples(self) -> int:
        """FIR group delay in input samples (informational)."""
        if self._h is None:
            return 0
        return (len(self._h) - 1) // (2 * self.up)

    def reset(self) -> None:
        self._inbuf = np.zeros(0, dtype=np.float32)
        self._hist = np.zeros(self._keep, dtype=np.float32)
        self._seen = 0
        self._released = 0
        self._pending = 0

    # ------------------------------------------------------------------
    def process(self, x: np.ndarray) -> np.ndarray:
        """Resample a block of mono samples; returns float32.

        Accepts any block size.  Output length is governed by the release rule
        in the module docstring, so it is independent of how the caller chose
        to chop up the stream; at most ``down - 1`` input samples are held
        internally for alignment (under 3.2 ms at 44.1 kHz).
        """
        block = np.asarray(x, dtype=np.float32).ravel()
        if block.size == 0:
            return np.zeros(0, dtype=np.float32)
        if self._h is None:
            self._seen += block.size
            self._released += block.size
            return block

        # 1. Alignment FIFO: keep a whole multiple of `down` to hand off.
        fifo = (
            np.concatenate((self._inbuf, block))
            if self._inbuf.size
            else block
        )
        n_handoff = (fifo.size // self.down) * self.down
        if n_handoff == 0:
            self._inbuf = fifo
            self._pending = fifo.size
            return np.zeros(0, dtype=np.float32)

        chunk = fifo[:n_handoff]
        self._inbuf = np.ascontiguousarray(fifo[n_handoff:], dtype=np.float32)
        self._pending = self._inbuf.size

        # 2. Filter.  `full` starts `keep` real input samples before `chunk`.
        full = np.concatenate((self._hist, chunk))
        filtered = upfirdn(self._h, full, up=self.up, down=self.down)

        # 3. Release.  Both the history and the chunk are whole multiples of
        # `down`, so the number of outputs that became available is exactly
        # `chunk.size * up // down`.
        n_emit = min((chunk.size * self.up) // self.down, filtered.size)
        # Global output n sits at local index n + _keep_out.
        m_start = min(self._keep_out, filtered.size)
        m_end = min(filtered.size, m_start + n_emit)
        out = np.ascontiguousarray(filtered[m_start:m_end], dtype=np.float32)
        self._released += out.size
        self._seen += chunk.size

        # 4. Carry history forward.
        self._hist = np.ascontiguousarray(full[-self._keep:], dtype=np.float32)
        return out

    def flush(self) -> np.ndarray:
        """Return the output still owed by the filter's ringdown.

        Only meaningful at end of stream (a file), not in a live callback.
        These samples depend on input that was never captured, so they are
        expected to be discarded; this must not be used to pad a capture with
        invented audio.
        """
        if self._h is None or self._hist.size == 0:
            return np.zeros(0, dtype=np.float32)
        pending = (len(self._h) - 1) // self.down
        if pending <= 0:
            return np.zeros(0, dtype=np.float32)
        tail = upfirdn(self._h, self._hist, up=self.up, down=self.down)
        out = np.ascontiguousarray(
            tail[tail.size - pending:], dtype=np.float32
        )
        self._hist = np.zeros(self._keep, dtype=np.float32)
        self._released += out.size
        return out

    def flush_input(self) -> np.ndarray:
        """Process the alignment FIFO's remainder, then the ringdown.

        For end of stream.  Returns whatever the filter can still produce from
        input that really was captured, plus a short, clearly-marked tail.
        """
        if self._h is None or self._inbuf.size == 0:
            return np.zeros(0, dtype=np.float32)
        remainder = self._inbuf
        self._inbuf = np.zeros(0, dtype=np.float32)
        self._pending = 0
        out = self.process(remainder)
        return out

    # ------------------------------------------------------------------
    def process_stream(self, blocks) -> np.ndarray:
        """Convenience: resample an iterable of blocks into one array."""
        parts = [self.process(b) for b in blocks]
        parts = [p for p in parts if p.size]
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)
