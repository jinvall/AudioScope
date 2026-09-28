"""Continuous raw recording.

docs/DATA_AND_STORAGE.md section 6: chunked WAV files, 15 minutes each by
default.  The recording is independent of event processing - it is the
uninterrupted evidence stream, and nothing downstream may alter it
(AGENTS.md section 2.1, section 22).

Disk I/O happens on a worker thread.  The capture side only ever does a
non-blocking queue put, and if the queue is full the overrun is *counted and
reported* rather than silently ignored: pretending a continuous recording is
continuous when it is not would corrupt the evidence chain.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from ..config import AppConfig
from .wavio import WavWriter


@dataclass
class RecorderStats:
    """Observable record of what the recorder actually did."""

    chunks_written: list[str] = field(default_factory=list)
    frames_written: int = 0
    blocks_dropped: int = 0
    frames_dropped: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        return self.frames_written / 48000.0 if self.frames_written else 0.0

    @property
    def is_contiguous(self) -> bool:
        """False if any audio was lost.  Must not be hidden."""
        return self.blocks_dropped == 0 and self.frames_dropped == 0

    def to_dict(self) -> dict:
        return {
            "chunks_written": list(self.chunks_written),
            "chunk_count": len(self.chunks_written),
            "frames_written": self.frames_written,
            "blocks_dropped": self.blocks_dropped,
            "frames_dropped": self.frames_dropped,
            "is_contiguous": self.is_contiguous,
            "errors": list(self.errors),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
        }


class Recorder:
    """Writes the capture stream to chunked WAV files on a worker thread."""

    def __init__(
        self,
        config: AppConfig,
        directory: Optional[str] = None,
        queue_blocks: Optional[int] = None,
    ) -> None:
        self.config = config
        self.sample_rate = config.sample_rate
        self.directory = os.path.abspath(
            directory or config.record.directory
        )
        self._queue: queue.Queue = queue.Queue(
            maxsize=queue_blocks or config.audio.queue_max_blocks
        )
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._writer: Optional[WavWriter] = None
        self.stats = RecorderStats()
        # Sentinel tells the worker to finish and write the final chunk.
        self._SENTINEL = object()

    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, start_index: int = 0) -> None:
        if self.running:
            return
        os.makedirs(self.directory, exist_ok=True)
        self._stop.clear()
        self._writer = WavWriter(
            directory=self.directory,
            sample_rate=self.sample_rate,
            chunk_frames=int(self.config.record.chunk_seconds * self.sample_rate),
            start_index=start_index,
        )
        self.stats = RecorderStats(started_at=datetime.now())
        self._thread = threading.Thread(
            target=self._run, name="recorder", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------
    @property
    def chunk_seconds(self) -> float:
        """The chunk length in force, which a control request can change."""
        writer = self._writer
        if writer is None or not self.config.sample_rate:
            return 0.0
        return writer.chunk_frames / float(self.config.sample_rate)

    def set_chunk_seconds(self, seconds: float) -> bool:
        """Change the chunk length for the *next* chunk.  Returns True if applied.

        The chunk being written now is not cut short and not extended: it
        finishes at the length it started with, and the new length takes effect
        when the writer next rolls over.  Anything else would either truncate
        a recording that is in progress or silently keep the old value until
        the next restart, and both are worse than waiting for the next file.

        ``WavWriter`` reads ``chunk_frames`` on every append, so this is a
        single attribute write and the writer thread picks it up immediately.
        """
        from ..control import MAX_CHUNK_SECONDS, MIN_CHUNK_SECONDS

        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            return False
        if not (MIN_CHUNK_SECONDS <= seconds <= MAX_CHUNK_SECONDS):
            return False
        writer = self._writer
        if writer is None:
            return False
        writer.chunk_frames = max(1, int(seconds * self.config.sample_rate))
        return True

    def write(self, samples: np.ndarray) -> bool:
        """Hand a block to the writer thread.  Never blocks.

        Returns False if the block had to be dropped because the disk could
        not keep up.  The caller is expected to surface that.
        """
        block = np.asarray(samples, dtype=np.float32).ravel()
        if block.size == 0:
            return True
        try:
            self._queue.put_nowait(block)
            return True
        except queue.Full:
            self.stats.blocks_dropped += 1
            self.stats.frames_dropped += int(block.size)
            return False

    def _run(self) -> None:
        writer = self._writer
        assert writer is not None
        while True:
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            if item is self._SENTINEL:
                break
            try:
                writer.append(item)
                self.stats.frames_written += int(item.size)
                self.stats.chunks_written = list(writer.chunks_written)
            except Exception as exc:
                # A disk failure must be recorded, not swallowed: the
                # recording is the primary evidence.
                self.stats.errors.append(f"{type(exc).__name__}: {exc}")
        try:
            writer.close()
            self.stats.chunks_written = list(writer.chunks_written)
        except Exception as exc:
            self.stats.errors.append(f"close failed: {type(exc).__name__}: {exc}")
        self.stats.stopped_at = datetime.now()

    # ------------------------------------------------------------------
    def stop(self, timeout: float = 10.0) -> RecorderStats:
        """Drain the queue, close the final chunk, and join the worker."""
        self._stop.set()
        try:
            self._queue.put_nowait(self._SENTINEL)
        except queue.Full:
            # The worker will see _stop and exit on its own timeout path.
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                self.stats.errors.append("recorder thread did not stop in time")
            self._thread = None
        return self.stats

    def __enter__(self) -> "Recorder":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


# ----------------------------------------------------------------------
class FileSource:
    """Replays a WAV file as if it were a live capture source.

    docs/AUDIO_PIPELINE.md section 13 requires file input to enter the *same*
    analysis path as live input.  Feeding a file through this source rather
    than a parallel code path is what keeps the test harness honest - it cannot
    drift from the live pipeline because it is the live pipeline.
    """

    def __init__(self, path: str, config: AppConfig, realtime: bool = False) -> None:
        from .wavio import probe_wav, read_wav

        self.path = os.path.abspath(path)
        self.config = config
        self.realtime = realtime
        self._info = probe_wav(self.path)
        self._data = read_wav(self.path, 0, None, mono=True)
        self._position = 0
        self._block = config.audio.block_size
        self._stop = threading.Event()
        self._name = os.path.basename(self.path)

    @property
    def name(self) -> str:
        return f"file:{self._name}"

    @property
    def native_sample_rate(self) -> int:
        return self._info.sample_rate

    @property
    def output_device(self) -> str:
        return f"file {self._name} @ {self._info.sample_rate} Hz"

    @property
    def total_frames(self) -> int:
        return int(self._data.size)

    @property
    def duration_seconds(self) -> float:
        return self._info.duration

    @property
    def is_running(self) -> bool:
        return self._position < self._data.size and not self._stop.is_set()

    def start(self) -> None:
        self._stop.clear()

    def read(self, frames: Optional[int] = None) -> Optional[np.ndarray]:
        if self._stop.is_set():
            return None
        size = frames or self._block
        if self._position >= self._data.size:
            return None
        start = self._position
        end = min(start + size, self._data.size)
        block = self._data[self._position:end]
        self._position = end
        if self.realtime:
            # Pace to wall-clock so a file source exercises the same queue
            # depths and timing as a live device.
            time.sleep((end - start) / self._info.sample_rate)
        return np.ascontiguousarray(block, dtype=np.float32)

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> "FileSource":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
