"""The analysis worker.

docs/ARCHITECTURE.md section 3: analysis runs on its own thread, never on the
capture thread and never in the audio callback.  Priority order from
docs/PERFORMANCE.md section 1 is capture, then buffering, then analysis, then
UI, then separation - so when the machine is loaded, *this* is the stage that
should fall behind, and it must never be able to block the stages above it.

That is enforced structurally:

* :meth:`AnalysisWorker.submit` is called from the capture pump thread and does
  nothing but a non-blocking ``put`` on a bounded queue.  If analysis cannot
  keep up, blocks are dropped and counted; the audio itself is unaffected
  because the recorder has its own queue.
* Dropped blocks are reported, never hidden.  A gap in the analysis is visible
  as :attr:`AnalysisWorker.blocks_dropped` and as ``gap_samples`` on the
  affected result, so a detector downstream can tell that evidence is missing
  rather than concluding nothing happened.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..config import AppConfig
from .features import FeatureExtractor, FrameFeatures
from .frames import FrameExtractor as WindowingFrameExtractor
from .noise_floor import AdaptiveNoiseFloor, NoiseFloorState
from .spectrogram import SpectrogramBuffer


@dataclass(frozen=True)
class AnalysisResult:
    """Features plus the noise-floor state for one frame."""

    features: FrameFeatures
    floor: NoiseFloorState
    # Absolute stream sample index of the frame's first sample.
    start_sample: int
    # Wall-clock seconds since the worker started, for ordering only.
    elapsed: float
    # True when audio was dropped between the previous frame and this one, so
    # this frame is not contiguous with its neighbour.
    gap: bool = False

    @property
    def index(self) -> int:
        return self.features.index

    @property
    def audio_time(self) -> float:
        """Position on the stream's own timeline, in seconds.

        This is the time base the waveform, spectrogram and event timeline must
        all share (docs/AUDIO_PIPELINE.md section 11).  ``elapsed`` is wall
        clock and is only meaningful while streaming at real time; during
        offline analysis they differ by the processing speed, so anything
        positioned in time must use this.
        """
        return self.start_sample / self.features.sample_rate

    @property
    def rms_db(self) -> float:
        return self.features.rms_db

    @property
    def snr_db(self) -> float:
        return self.floor.overall_snr_db

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start_sample": self.start_sample,
            "audio_time": self.audio_time,
            "elapsed": self.elapsed,
            "gap": self.gap,
            "features": self.features.to_dict(),
            "noise_floor": self.floor.to_dict(),
        }


@dataclass
class AnalysisStats:
    """Observable counters for the UI and the performance requirements."""

    blocks_in: int = 0
    blocks_dropped: int = 0
    frames: int = 0
    gap_frames: int = 0
    events: int = 0
    last_block_at: Optional[float] = None
    _analysis_seconds: float = 0.0
    _audio_seconds: float = 0.0
    started: Optional[float] = None
    stopped: Optional[float] = None
    errors: list[str] = field(default_factory=list)

    @property
    def queue_depth(self) -> int:
        return 0  # filled in by the worker

    @property
    def analysis_seconds(self) -> float:
        return self._analysis_seconds

    @property
    def realtime_ratio(self) -> float:
        """Analysis CPU time per second of audio.

        docs/PERFORMANCE.md section 6.  Below 1.0 means analysis is cheaper
        than real time, so the machine has headroom left over for everything
        else.
        """
        if self._audio_seconds <= 0:
            return 0.0
        return self._analysis_seconds / self._audio_seconds

    @property
    def mean_frame_ms(self) -> float:
        if not self.frames:
            return 0.0
        return 1000.0 * self._analysis_seconds / self.frames

    def to_dict(self) -> dict:
        return {
            "blocks_in": self.blocks_in,
            "blocks_dropped": self.blocks_dropped,
            "frames": self.frames,
            "gap_frames": self.gap_frames,
            "events": self.events,
            "audio_seconds": round(self._audio_seconds, 3),
            "analysis_seconds": round(self._analysis_seconds, 4),
            "realtime_ratio": round(self.realtime_ratio, 5),
            "mean_frame_ms": round(self.mean_frame_ms, 4),
            "is_contiguous": self.blocks_dropped == 0,
            "errors": list(self.errors),
        }


_SENTINEL = object()


class AnalysisWorker:
    """Consumes capture blocks and produces analysis results on a worker thread."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        result_queue_size: int = 256,
        input_queue_blocks: int = 128,
        on_result: Optional[Callable[[AnalysisResult], None]] = None,
        lossless: bool = False,
        history_size: int = 0,
    ) -> None:
        self.config = (config or AppConfig()).validate()
        # In lossless mode submit() waits for room instead of dropping.  That is
        # right for offline file analysis, where there is no real-time deadline
        # to protect and a third of the file being silently skipped would make
        # the result meaningless.  It is wrong for live capture, where dropping
        # analysis is always preferable to stalling the audio path.
        self.lossless = bool(lossless)
        # Optional retained history.  The live result queue is deliberately
        # small so a slow GUI cannot make the worker back up, which means it
        # cannot answer "what happened 10 seconds ago" on its own.  Offline
        # analysis needs the whole run, so it can opt into a larger buffer.
        self._history: deque[AnalysisResult] = deque(
            maxlen=history_size if history_size > 0 else 1
        )
        self._history_enabled = history_size > 0
        self.history_dropped = 0
        sr = self.config.sample_rate
        analysis = self.config.analysis

        self.sample_rate = sr
        self.frame_samples = analysis.frame_samples_at(sr)
        self.hop_samples = analysis.hop_samples_at(sr)
        self.frame_rate = sr / self.hop_samples

        # Frames are handed over unwindowed; the feature extractor windows
        # internally so level measurements stay on the raw samples.
        self._framer = WindowingFrameExtractor(self.frame_samples, self.hop_samples, None)
        self._features = FeatureExtractor(
            sr,
            rolloff_percentile=analysis.rolloff_percentile,
            window=analysis.window,
        )
        self._floor = AdaptiveNoiseFloor(analysis, frame_rate=self.frame_rate)
        self.spectrogram = SpectrogramBuffer(
            bins=self._features_n_bins(),
            columns=self._spectrogram_columns(),
            max_hz=min(24000.0, sr / 2.0),
        )

        self._in: queue.Queue = queue.Queue(maxsize=input_queue_blocks)
        self._out: queue.Queue = queue.Queue(maxsize=result_queue_size)
        self._on_result = on_result
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started_at = 0.0

        self.stats = AnalysisStats()
        # Absolute stream index expected next, used to detect dropped audio.
        self._expected_start = 0
        self._have_clock = False
        self._pending_gap = False

    # ------------------------------------------------------------------
    def _features_n_bins(self) -> int:
        # One column per usable part of the rfft output.
        return self.frame_samples // 2 + 1

    def _spectrogram_columns(self) -> int:
        # Enough columns for the ring buffer's worth of time at the analysis
        # rate, capped so a long buffer cannot create a huge image.
        seconds = min(self.config.buffer.seconds, 30.0)
        return max(64, int(seconds * self.frame_rate))

    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def input_queue_depth(self) -> int:
        return self._in.qsize()

    @property
    def result_queue_depth(self) -> int:
        return self._out.qsize()

    @property
    def noise_floor_db(self) -> float:
        return self._floor.overall_floor_db

    def latest(self) -> Optional[AnalysisResult]:
        """Most recent result, or None.  Does not consume the queue."""
        try:
            return self._out.queue[-1]
        except IndexError:
            return None

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._started_at = time.monotonic()
        self.stats = AnalysisStats(started=time.monotonic())
        self._thread = threading.Thread(
            target=self._run, name="analysis", daemon=True
        )
        self._thread.start()

    def submit(
        self, samples: np.ndarray, start_sample: Optional[int] = None
    ) -> bool:
        """Hand a capture block to the worker.  Never blocks.

        Returns False if the block had to be dropped.  Callers on the capture
        path should treat that as "analysis is behind", not as an audio error:
        the audio itself has already been recorded.
        """
        block = np.asarray(samples, dtype=np.float32).ravel()
        if block.size == 0:
            return True
        if not self.running:
            return False
        payload = (block, start_sample)
        if self.lossless:
            # Bounded by construction: the queue size still caps memory, and
            # stop() drains it, so this cannot deadlock on shutdown.
            while not self._stop.is_set():
                try:
                    self._in.put(payload, timeout=0.1)
                    self.stats.blocks_in += 1
                    return True
                except queue.Full:
                    continue
            return False
        try:
            self._in.put_nowait(payload)
            self.stats.blocks_in += 1
            return True
        except queue.Full:
            # Drop the oldest so the newest audio is the one analysed: for a
            # live monitor, what just happened matters more than what was
            # queued a moment ago.
            try:
                self._in.get_nowait()
            except queue.Empty:
                pass
            try:
                self._in.put_nowait(payload)
            except queue.Full:
                self.stats.blocks_dropped += 1
                self._pending_gap = True
                return False
            self.stats.blocks_dropped += 1
            self._pending_gap = True
            return False

    # ------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            try:
                item = self._in.get(timeout=0.2)
            except queue.Empty:
                # Nothing pending.  Only give up if we have also been asked to
                # stop and the sentinel could not be queued because the input
                # was full.
                if self._stop.is_set():
                    break
                continue
            if item is _SENTINEL:
                break
            block, start_sample = item
            try:
                self._process(block, start_sample)
            except Exception as exc:
                # An analysis failure must never take the capture path down
                # (docs/ARCHITECTURE.md section 9).
                self.stats.errors.append(
                    f"{type(exc).__name__}: {exc}"
                )
                if len(self.stats.errors) > 50:
                    del self.stats.errors[:-50]

    def _process(
        self, block: np.ndarray, start_sample: Optional[int]
    ) -> None:
        began = time.perf_counter()
        self.stats.last_block_at = time.monotonic()

        # Establish or check the sample clock.
        gap = self._pending_gap
        self._pending_gap = False
        if start_sample is not None:
            if self._have_clock and start_sample > self._expected_start:
                gap = True
            self._expected_start = start_sample
            self._have_clock = True

        block_start = self._expected_start
        self._expected_start += int(block.size)
        self.stats._audio_seconds += block.size / self.sample_rate

        frames = self._framer.push(block)
        if not frames:
            return

        for offset, frame in enumerate(frames):
            features = self._features.analyse(
                frame, block_start + offset * self.hop_samples
            )
            self.spectrogram.push(self._features_magnitude())
            floor = self._floor.update(features)
            result = AnalysisResult(
                features=features,
                floor=floor,
                start_sample=block_start + offset * self.hop_samples,
                elapsed=time.monotonic() - self._started_at,
                gap=gap,
            )
            self._publish(result)

        self.stats.frames += len(frames)
        if gap:
            self.stats.gap_frames += 1
        self.stats._analysis_seconds += time.perf_counter() - began

    def _features_magnitude(self) -> np.ndarray:
        """Magnitude spectrum for the spectrogram column.

        Taken from the feature extractor's own transform rather than running a
        second rfft over the same frame.  Both need the same windowed spectrum,
        so the second transform was pure duplicate work at 100 frames per
        second.
        """
        magnitude = self._features.last_magnitude
        if magnitude is None or magnitude.size != self.spectrogram.bins:
            return np.zeros(self.spectrogram.bins, dtype=np.float32)
        return magnitude

    # ------------------------------------------------------------------
    def _publish(self, result: AnalysisResult) -> None:
        try:
            self._out.put_nowait(result)
        except queue.Full:
            # A slow consumer (the GUI) must not stall analysis.  Drop the
            # oldest result and keep the newest, which is what a live view
            # wants.
            try:
                self._out.get_nowait()
                self._out.put_nowait(result)
            except (queue.Empty, queue.Full):
                pass
        self.stats.events += 1
        if self._history_enabled:
            if len(self._history) == self._history.maxlen:
                self.history_dropped += 1
            self._history.append(result)
        if self._on_result is not None:
            try:
                self._on_result(result)
            except Exception as exc:
                self.stats.errors.append(f"on_result: {type(exc).__name__}: {exc}")

    def results(self, limit: Optional[int] = None) -> list[AnalysisResult]:
        """Drain up to ``limit`` queued results.

        Drains the *live* queue, which is small by design.  For a complete
        offline record use :attr:`history` instead.
        """
        out: list[AnalysisResult] = []
        while limit is None or len(out) < limit:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                break
        return out

    @property
    def history(self) -> list[AnalysisResult]:
        """Retained results, oldest first.  Empty unless history was enabled."""
        if not self._history_enabled:
            return []
        return list(self._history)

    @property
    def history_size(self) -> int:
        return self._history.maxlen if self._history_enabled else 0

    # ------------------------------------------------------------------
    def stop(self, timeout: float = 10.0) -> AnalysisStats:
        """Stop the worker, drain the input queue, and join."""
        self._stop.set()
        try:
            self._in.put_nowait(_SENTINEL)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                self.stats.errors.append("analysis thread did not stop in time")
            self._thread = None
        self.stats.stopped = time.monotonic()
        return self.stats

    def __enter__(self) -> "AnalysisWorker":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
