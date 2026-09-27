"""Audio source abstraction.

Phase 1 has two ways to get audio in:

* :class:`DeviceCapture` - a local microphone, via PortAudio.
* :class:`NetworkSource` - a TCP stream from an Android device, on port 8090
  (see :mod:`app.audio.network`).

Both emit the *internal* format from :mod:`app.config`: mono, float32, 48 kHz.
Anything the device or the sender actually provides is converted once, at this
boundary, so nothing downstream ever has to care.

The source's contract with the rest of the pipeline is deliberately small:
:meth:`AudioSource.read` returns the next block of mono float32 at the
internal rate, or ``None`` at end of stream.  A source never does analysis,
separation, disk I/O or GUI work.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

from ..config import AppConfig
from .devices import AudioDeviceError, can_open, resolve_device
from .resample import StreamingResampler, ratio_for


class AudioSource(ABC):
    """Common interface for live audio inputs."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.sample_rate = config.sample_rate
        self._stop = threading.Event()

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable description, recorded in event metadata."""

    @property
    @abstractmethod
    def native_sample_rate(self) -> int:
        """Rate the source actually delivers before internal conversion."""

    @property
    def is_running(self) -> bool:
        return not self._stop.is_set()

    @abstractmethod
    def start(self) -> None:
        """Begin delivering audio.  Returns as soon as capture is live."""

    @abstractmethod
    def read(
        self, frames: Optional[int] = None, timeout: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """Next block of mono float32, or None when the source has ended.

        ``frames`` is advisory: a live device delivers whatever PortAudio's
        callback provides, so implementations may ignore it.  ``timeout`` is
        optional: a source that blocks until audio arrives may ignore it, and
        one that supports polling will return None once it expires.
        """

    def stop(self) -> None:
        """Signal the source to finish.  Safe to call more than once."""
        self._stop.set()

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> "AudioSource":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    def _make_resampler(self, source_rate: int) -> Optional[StreamingResampler]:
        """Build a converter if the source rate differs from the internal one.

        Returns None when no conversion is needed, so the common case does no
        filtering at all.
        """
        if source_rate == self.sample_rate:
            return None
        up, down = ratio_for(source_rate, self.sample_rate)
        return StreamingResampler(up, down)

    @staticmethod
    def _to_mono(block: np.ndarray) -> np.ndarray:
        """Average a (frames, channels) block down to mono float32.

        This is the single place multi-channel input becomes mono, per
        AGENTS.md section 6.  Averaging preserves correlated content instead of
        discarding a channel.
        """
        if block.ndim == 1:
            return np.ascontiguousarray(block, dtype=np.float32)
        return np.ascontiguousarray(block.mean(axis=1), dtype=np.float32)


# ----------------------------------------------------------------------
# Local device
# ----------------------------------------------------------------------
class DeviceCapture(AudioSource):
    """Continuous capture from a local input device.

    The PortAudio callback is the real-time thread: it only converts to the
    internal format and hands the block to a bounded queue.  Everything else
    - disk writes, analysis, GUI - happens elsewhere (AGENTS.md section 2.3).
    """

    def __init__(
        self,
        config: AppConfig,
        device: str | int | None = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(config)
        self._selector = device if device is not None else config.audio.device
        self._device_info = resolve_device(self._selector)
        self._requested_rate = config.sample_rate
        self._device = self._device_info.index
        self._on_error = on_error
        self._stream = None
        self._resampler: Optional[StreamingResampler] = None
        self._rate_in_use = self._requested_rate
        self._blocks: list[np.ndarray] = []
        self._cond = threading.Condition()
        self._opened_at_rate: Optional[int] = None
        # PortAudio reports an input overflow when its buffer is not serviced in
        # time, which means audio was genuinely dropped.  That is a loss of
        # evidence, so it is counted and reported rather than only printed.
        self.input_overflows = 0
        self.overflow_messages: list[str] = []

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._device_info.name

    @property
    def device_info(self):
        return self._device_info

    @property
    def native_sample_rate(self) -> int:
        return self._rate_in_use

    @property
    def is_converting(self) -> bool:
        """True when the device rate had to be converted to the internal rate."""
        return self._resampler is not None

    @property
    def output_device(self) -> str:
        """The device that actually supplies audio, for metadata."""
        return f"{self._device_info.name} (index {self._device})"

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Open the device and begin delivering audio.

        Idempotent: an already-open stream is left alone, so a caller may
        start the source and then hand it to the pipeline, which calls
        ``start()`` again.
        """
        if self._stream is not None:
            return
        import sounddevice as sd

        self._stop.clear()
        rate, self._resampler = self._open_at_best_rate(sd)
        self._rate_in_use = rate

        try:
            self._stream = sd.InputStream(
                device=self._device,
                channels=1,
                samplerate=rate,
                dtype="float32",
                blocksize=self.config.audio.block_size,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise AudioDeviceError(
                f"could not open input device {self._device_info.name!r}: {exc}"
            ) from exc

    def _open_at_best_rate(self, sd) -> tuple[int, Optional[StreamingResampler]]:
        """Prefer the internal rate; fall back to the device's own rate.

        docs/AUDIO_PIPELINE.md section 1: convert once, at the input boundary.
        Using the internal rate directly when the device allows it avoids the
        conversion altogether, which is both cheaper and better quality.
        """
        if can_open(self._device, self._requested_rate, 1):
            return self._requested_rate, None

        native = int(round(self._device_info.default_sample_rate))
        if can_open(self._device, native, 1):
            return native, self._make_resampler(native)

        # Last resort: ask PortAudio for anything it will give us mono.
        for candidate in (48000, 44100, 32000, 16000, 8000):
            if can_open(self._device, candidate, 1):
                return candidate, self._make_resampler(candidate)
        raise AudioDeviceError(
            f"device {self._device_info.name!r} cannot be opened for mono input "
            f"at 48 kHz, {native} Hz, or any common fallback rate"
        )

    # ------------------------------------------------------------------
    def _callback(self, indata, frames, time_info, status) -> None:
        """PortAudio callback.  Runs on the real-time thread.

        Deliberately minimal: one conversion and one queue append.  No disk
        I/O, no analysis, no allocation beyond the block PortAudio already
        gave us, and no blocking (AGENTS.md section 2.3, PERFORMANCE.md 2).
        """
        if status:
            message = str(status)
            if "overflow" in message.lower():
                self.input_overflows += 1
                if len(self.overflow_messages) < 20:
                    self.overflow_messages.append(message)
            if self._on_error:
                self._on_error(f"capture status: {message}")

        block = indata
        if block.ndim > 1 and block.shape[1] > 1:
            block = block.mean(axis=1)
        block = np.ascontiguousarray(block, dtype=np.float32)

        if self._resampler is not None:
            block = self._resampler.process(block)

        with self._cond:
            self._blocks.append(block)
            self._cond.notify_all()

    # ------------------------------------------------------------------
    def read(self, frames: Optional[int] = None) -> Optional[np.ndarray]:
        """Next available block, waiting briefly for one to arrive.

        Returns None once the source has been stopped and drained, so callers
        can treat end of stream as a normal outcome.
        """
        with self._cond:
            while not self._blocks:
                if self._stop.is_set():
                    return None
                self._cond.wait(timeout=0.5)
            return self._blocks.pop(0)

    def drain(self) -> list[np.ndarray]:
        """Take everything currently queued without waiting."""
        with self._cond:
            blocks = self._blocks
            self._blocks = []
            return blocks

    # ------------------------------------------------------------------
    def stop(self) -> None:
        super().stop()
        with self._cond:
            self._cond.notify_all()

    def close(self) -> None:
        self.stop()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                # Closing a device that PortAudio already tore down is not an
                # error worth propagating during shutdown.
                pass
