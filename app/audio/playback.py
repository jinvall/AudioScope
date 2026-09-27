"""Audio playback.

tasks/PHASE_01_AUDIO.md task 4 and AGENTS.md section 18: play, pause, stop,
seek, loop, volume.  Playback must use real captured or generated audio
(docs/AUDIO_PIPELINE.md section 12) - there is no synthesised stand-in here.

The stream callback pulls from the in-memory buffer under a short lock and
multiplies by the volume.  It does no file I/O, so the GUI thread stays free.
"""

from __future__ import annotations

import threading
from enum import Enum
from typing import Optional

import numpy as np
import sounddevice as sd


class PlaybackState(Enum):
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"

    @property
    def is_active(self) -> bool:
        return self is PlaybackState.PLAYING


class PlaybackError(RuntimeError):
    pass


class AudioPlayer:
    """Plays mono float32 buffers through a real output device.

    Typical use::

        with AudioPlayer() as player:
            player.load(np_array, 48000)
            player.play()
            player.seek(2.0)
    """

    def __init__(
        self,
        sample_rate: int = 48_000,
        device: str | int | None = None,
        block_size: int = 1024,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self._device = device

        self._data = np.zeros(0, dtype=np.float32)
        self._position = 0  # in frames
        self._volume = 1.0
        self._loop = False
        self._state = PlaybackState.STOPPED
        self._lock = threading.RLock()
        self._stream: Optional[sd.OutputStream] = None
        self._finished = threading.Event()
        self._finished.set()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def state(self) -> PlaybackState:
        with self._lock:
            return self._state

    @property
    def volume(self) -> float:
        with self._lock:
            return self._volume

    @property
    def loop(self) -> bool:
        with self._lock:
            return self._loop

    @property
    def frames_loaded(self) -> int:
        with self._lock:
            return int(self._data.size)

    @property
    def duration_seconds(self) -> float:
        with self._lock:
            return self._data.size / self.sample_rate if self.sample_rate else 0.0

    @property
    def position_seconds(self) -> float:
        with self._lock:
            return self._position / self.sample_rate

    @property
    def is_playing(self) -> bool:
        return self.state is PlaybackState.PLAYING

    @property
    def is_loaded(self) -> bool:
        return self.frames_loaded > 0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self, samples: np.ndarray, sample_rate: Optional[int] = None) -> None:
        """Replace the loaded audio and stop playback."""
        if sample_rate is not None and int(sample_rate) != self.sample_rate:
            # Resampling playback is a nicety, not a requirement: the whole
            # pipeline emits the internal rate, so a mismatch means a caller
            # bug.  Refuse rather than silently play at the wrong speed.
            raise PlaybackError(
                f"player runs at {self.sample_rate} Hz but was given "
                f"{sample_rate} Hz audio; resample to the internal rate first"
            )
        with self._lock:
            self._data = np.ascontiguousarray(
                np.asarray(samples, dtype=np.float32).ravel()
            )
            self._position = 0
            self._state = PlaybackState.STOPPED
        self._finished.set()

    def load_file(self, path: str) -> None:
        """Load a WAV file, converting it to the internal rate if needed."""
        from .resample import StreamingResampler, ratio_for
        from .wavio import probe_wav, read_wav

        info = probe_wav(path)
        data = read_wav(path, 0, None, mono=True)
        if info.sample_rate != self.sample_rate:
            up, down = ratio_for(info.sample_rate, self.sample_rate)
            data = StreamingResampler(up, down).process(data)
        self.load(data, self.sample_rate)

    def unload(self) -> None:
        self.stop()
        with self._lock:
            self._data = np.zeros(0, dtype=np.float32)
            self._position = 0

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------
    def play(self) -> None:
        """Start or resume playback from the current position."""
        with self._lock:
            if self._data.size == 0:
                raise PlaybackError("nothing loaded to play")
            if self._state is PlaybackState.PLAYING:
                return
            if self._position >= self._data.size:
                self._position = 0
            self._state = PlaybackState.PLAYING
        self._finished.clear()
        self._ensure_stream()

    def pause(self) -> None:
        with self._lock:
            if self._state is not PlaybackState.PLAYING:
                return
            self._state = PlaybackState.PAUSED
        self._finish_stream()
        self._finished.set()

    def stop(self) -> None:
        """Stop and rewind to the start."""
        with self._lock:
            self._state = PlaybackState.STOPPED
            self._position = 0
        self._finish_stream()
        self._finished.set()

    def seek(self, seconds: float) -> None:
        """Move the play head.  Accepts negative values (clamped to 0)."""
        with self._lock:
            target = int(round(seconds * self.sample_rate))
            target = max(0, min(target, self._data.size))
            self._position = target
            if self._state is PlaybackState.PLAYING:
                self._finished.clear()

    def seek_relative(self, delta_seconds: float) -> None:
        self.seek(self.position_seconds + delta_seconds)

    def set_volume(self, volume: float) -> None:
        v = float(volume)
        if not 0.0 <= v <= 1.0:
            raise PlaybackError(f"volume must be 0.0-1.0, got {volume}")
        with self._lock:
            self._volume = v

    def set_loop(self, enabled: bool) -> None:
        with self._lock:
            self._loop = bool(enabled)

    def wait_until_finished(self, timeout: Optional[float] = None) -> bool:
        """Block until playback reaches the end.  False on timeout."""
        return self._finished.wait(timeout)

    # ------------------------------------------------------------------
    # Stream plumbing
    # ------------------------------------------------------------------
    def _ensure_stream(self) -> None:
        if self._stream is not None:
            if not self._stream.active:
                try:
                    self._stream.start()
                except Exception:
                    self._stream = None
            if self._stream is not None:
                return
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            callback=self._callback,
            device=self._device,
        )
        self._stream.start()

    def _finish_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def _callback(self, outdata, frames, time_info, status) -> None:
        """Output callback.  Renders already-loaded memory; no I/O."""
        buffer = outdata[:, 0]
        with self._lock:
            if self._state is not PlaybackState.PLAYING or self._data.size == 0:
                buffer[:] = 0.0
                return

            start = self._position
            end = start + frames

            if end > self._data.size:
                if self._loop:
                    # Tile the buffer to fill the block without a click at the
                    # wrap point being avoidable: the samples are contiguous
                    # in memory, so copy in loop-sized pieces.
                    written = 0
                    pos = start % self._data.size
                    while written < frames:
                        take = min(frames - written, self._data.size - pos)
                        buffer[written:written + take] = self._data[pos:pos + take]
                        written += take
                        pos = 0 if pos + take >= self._data.size else pos + take
                    self._position = (start + frames) % self._data.size
                else:
                    available = max(0, self._data.size - start)
                    if available:
                        buffer[:available] = self._data[start:start + available]
                    buffer[available:] = 0.0
                    self._position = self._data.size
                    self._state = PlaybackState.STOPPED
                    self._finished.set()
                    if self._volume != 1.0:
                        buffer *= self._volume
                    return
            else:
                buffer[:] = self._data[start:end]
                self._position = end
                if self._position >= self._data.size and not self._loop:
                    self._state = PlaybackState.STOPPED
                    self._finished.set()

            if self._volume != 1.0:
                buffer *= self._volume

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.stop()
        self.unload()

    def __enter__(self) -> "AudioPlayer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
