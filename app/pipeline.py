"""The audio pipeline: source -> ring buffer -> recorder.

This is the single path every input goes through.  Live capture and file
playback differ only in which :class:`~app.audio.capture.AudioSource` is
supplied (docs/AUDIO_PIPELINE.md section 13), so the test harness exercises
the real pipeline instead of a parallel imitation of it.

Phase 1 stops here on purpose.  Analysis, event detection and separation are
later phases; the pipeline exposes the hook they will attach to
(:attr:`AudioPipeline.on_block`) without implementing any of them, so there is
no placeholder pretending to be a detector.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .audio.capture import AudioSource
from .audio.recorder import Recorder, RecorderStats
from .audio.ringbuffer import RingBuffer
from .config import AppConfig


@dataclass
class PipelineStats:
    """Live counters for the UI and for the performance requirements."""

    blocks: int = 0
    frames: int = 0
    started: Optional[float] = None
    stopped: Optional[float] = None
    overruns: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def audio_seconds(self) -> float:
        return self.frames / 48000.0 if self.frames else 0.0

    @property
    def wall_seconds(self) -> float:
        if self.started is None:
            return 0.0
        end = self.stopped if self.stopped is not None else time.monotonic()
        return max(0.0, end - self.started)

    @property
    def realtime_ratio(self) -> float:
        """processing_time / audio_duration; below 1.0 keeps up with realtime.

        docs/PERFORMANCE.md section 6.
        """
        audio = self.audio_seconds
        if audio <= 0:
            return 0.0
        return self.wall_seconds / audio

    def to_dict(self) -> dict:
        return {
            "blocks": self.blocks,
            "frames": self.frames,
            "audio_seconds": round(self.audio_seconds, 3),
            "wall_seconds": round(self.wall_seconds, 3),
            "realtime_ratio": round(self.realtime_ratio, 3),
            "ring_overruns": self.overruns,
            "errors": list(self.errors),
        }


class AudioPipeline:
    """Drives a source into the ring buffer and the recorder."""

    def __init__(
        self,
        config: AppConfig,
        source: AudioSource,
        record: bool = True,
        record_directory: Optional[str] = None,
    ) -> None:
        self.config = config
        self.source = source
        self.ring = RingBuffer(config.sample_rate, config.buffer.seconds)
        self.recorder = (
            Recorder(config, directory=record_directory) if record else None
        )
        self.stats = PipelineStats()
        # Consumers attach here.  ``enable_analysis`` chains itself onto
        # whatever is already registered so an embedding application keeps its
        # own hook.
        self.on_block: Optional[Callable[[int, np.ndarray], None]] = None
        self._analysis: Optional["AnalysisWorker"] = None
        self._detection: Optional["DetectionWorker"] = None
        self._writer = None
        self._retention = None
        self._user_hook: Optional[Callable[[int, np.ndarray], None]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._finished.set()

    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def recorder_stats(self) -> Optional[RecorderStats]:
        return self.recorder.stats if self.recorder else None

    def enable_analysis(
        self, lossless: bool = False, history_size: int = 0
    ) -> "AnalysisWorker":
        """Attach an analysis worker and return it.

        The worker is started here and stopped by :meth:`stop`, so a caller
        that only wants analysis does not have to manage its lifetime.  The
        worker's ``submit`` is non-blocking, so attaching it cannot slow
        capture down (AGENTS.md section 2.3).
        """
        from .analysis.worker import AnalysisWorker

        if self._analysis is not None:
            return self._analysis
        worker = AnalysisWorker(
            self.config, lossless=lossless, history_size=history_size
        )
        worker.spectrogram.column_seconds = (
            self.config.analysis_hop_frames / self.config.sample_rate
        )
        self._user_hook = self.on_block

        def forward(position: int, block: np.ndarray) -> None:
            if self._user_hook is not None:
                self._user_hook(position, block)
            # A dropped block is counted by the worker.  It must not raise
            # here, because this runs on the capture pump.
            try:
                worker.submit(block, position)
            except Exception:
                pass

        self.on_block = forward
        self._analysis = worker
        if self.running:
            worker.start()
        return worker

    @property
    def analysis(self) -> Optional["AnalysisWorker"]:
        """The attached analysis worker, or None."""
        return self._analysis

    def enable_detection(
        self,
        on_event: Optional[Callable] = None,
        event_root: Optional[str] = None,
        lossless: bool = False,
        database: Optional[str] = None,
        persist: bool = True,
    ) -> "DetectionWorker":
        """Attach the analysis worker (if needed) and the detection stage.

        Detection consumes analysis results, so this attaches both.  The
        detection worker is fed by a non-blocking put from the analysis worker's
        result callback, so neither stage can stall the capture path.
        """
        from .detection.worker import DetectionWorker
        from .events.retention import RetentionManager
        from .events.store import EventStore

        if self._detection is not None:
            return self._detection
        analysis = self.analysis
        if analysis is None:
            analysis = self.enable_analysis(lossless=lossless)

        # Audio retention wraps the store, so every event write is accounted
        # and bounded.  It needs the database to know which class a stored
        # file belongs to when choosing what to evict, so the database is
        # opened first when persistence is in use.
        database_path = database
        db = None
        writer = None
        retention = None
        if persist and database_path is not None:
            from .events.database import EventDatabase
            from .events.persistence import EventWriter

            db = EventDatabase(database_path)
            retention = RetentionManager(
                self.config.audio_retention,
                database=db,
                root=os.path.abspath(event_root or self.config.output_dir),
            )
            # Startup recovery: load the persisted accounting, validate it
            # against the database, reconcile with the filesystem, and clear
            # up any classification already over its cap.  Done once, before
            # any capture, so the first event of a session is already
            # accounted for correctly.
            try:
                retention.load_accounting()
                retention.reconcile()
            except Exception:
                # A retention problem must never prevent the application from
                # starting; the caps are a storage policy, not a precondition
                # for listening.
                pass
            self._retention = retention

            # Persistence runs on its own thread, so a slow disk or a
            # database lock can never stall event extraction on the
            # detection thread.
            writer = EventWriter(db, retention=retention)
            writer.start()
            self._writer = writer

        store = EventStore(
            event_root or self.config.output_dir, retention=retention
        )
        worker = DetectionWorker(
            self.config, self.ring, store=store, on_event=on_event,
            writer=writer,
        )
        source = self.source
        worker.set_source_info(
            provider=lambda: _source_info_of(source),
        )

        existing = analysis._on_result

        def forward(result) -> None:
            if existing is not None:
                existing(result)
            worker.submit(result)

        analysis._on_result = forward
        self._detection = worker
        if self.running:
            worker.start()
        return worker

    @property
    def retention(self):
        """The attached audio-retention manager, or None.

        Exposed so a caller can read the live accounting and force a
        reconciliation; the hot path never needs it.
        """
        return self._retention

    @property
    def detection(self) -> Optional["DetectionWorker"]:
        """The attached detection worker, or None."""
        return self._detection

    def start(self) -> None:
        """Start the source, the recorder, and the pump thread."""
        if self.running:
            return
        self._stop.clear()
        self._finished.clear()
        self.stats = PipelineStats(started=time.monotonic())
        if self.recorder is not None:
            self.recorder.start()
        if self._analysis is not None and not self._analysis.running:
            self._analysis.start()
        if self._detection is not None and not self._detection.running:
            self._detection.start()
        self.source.start()
        self._thread = threading.Thread(
            target=self._run, name="pipeline", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------
    def _run(self) -> None:
        """Pump blocks from the source to the ring buffer and recorder.

        Lives on its own thread so the capture callback is never waiting on
        disk or on analysis (AGENTS.md section 2.3).
        """
        while not self._stop.is_set():
            block = self.source.read()
            if block is None:
                break
            self._deliver(block)

        # Drain anything the source queued before it reported end of stream.
        drain = getattr(self.source, "drain", None)
        if callable(drain):
            for block in drain():
                self._deliver(block)

        self.stats.stopped = time.monotonic()
        self._stop.set()
        self._finished.set()

    def _deliver(self, block: np.ndarray) -> None:
        data = np.asarray(block, dtype=np.float32).ravel()
        if data.size == 0:
            return
        self.ring.write(data)
        if self.recorder is not None and not self.recorder.write(data):
            self.stats.overruns += 1
        self.stats.blocks += 1
        self.stats.frames += int(data.size)
        if self.on_block is not None:
            try:
                self.on_block(self.ring.total_written, data)
            except Exception as exc:
                # A consumer failure must not take the capture path down.
                self.stats.errors.append(f"on_block: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    def stop(self, timeout: float = 15.0) -> PipelineStats:
        """Stop the pipeline and flush the recorder."""
        self._stop.set()
        source_stop = getattr(self.source, "stop", None)
        if callable(source_stop):
            source_stop()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                self.stats.errors.append("pipeline thread did not stop in time")
            self._thread = None
        close = getattr(self.source, "close", None)
        if callable(close):
            close()
        if self.recorder is not None:
            self.recorder.stop(timeout=timeout)
        if self._analysis is not None:
            self._analysis.stop(timeout=timeout)
        if self._detection is not None:
            # The ring buffer must still hold the last event's post-roll when
            # the detection stage drains, so detection stops after analysis.
            for event in self._detection.flush():
                pass
            self._detection.stop(timeout=timeout)
        # The writer owns its own thread and is deliberately stopped last, so
        # the events detection just handed it are drained into the index
        # before shutdown.  Without this the writer is a daemon thread and a
        # queued event can be lost at exit.
        writer = self._writer
        if writer is not None:
            writer.stop(timeout=timeout)
        if self.stats.stopped is None:
            self.stats.stopped = time.monotonic()
        self._finished.set()
        return self.stats

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the source reaches end of stream."""
        return self._finished.wait(timeout)

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """A snapshot for the CLI and the GUI status panel."""
        out = {
            "source": self.source.name,
            "device": getattr(self.source, "output_device", self.source.name),
            "sample_rate": self.config.sample_rate,
            "channels": 1,
            "ring_buffer_seconds": self.config.buffer.seconds,
            "ring_frames_held": self.ring.frames_available,
            "ring_overruns": self.ring.overruns,
            "pipeline": self.stats.to_dict(),
        }
        if self.recorder is not None:
            out["recorder"] = self.recorder.stats.to_dict()
        if self._analysis is not None:
            out["analysis"] = self._analysis.stats.to_dict()
        if self._detection is not None:
            out["detection"] = self._detection.stats.to_dict()
        return out


    def __enter__(self) -> "AudioPipeline":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def _source_info_of(source) -> dict:
    """Provenance for one event, read from the source at emit time."""
    return {
        "name": getattr(source, "name", ""),
        "stream_rate": getattr(source, "native_sample_rate", None),
        "wire_format": _wire_format_of(source),
        "client_configs": _client_configs_of(source),
    }


def _wire_format_of(source) -> Optional[str]:
    """Describe the wire format, when the source has one.

    Duck-typed on the network source's marker attribute rather than
    ``isinstance``: a wrapper or a test double around NetworkSource is still a
    network source, and reporting no wire format for it would quietly lose
    provenance rather than fail.
    """
    from .audio import network

    if hasattr(source, "client_configs"):
        return f"s16le {network.WIRE_SAMPLE_RATE} Hz mono"
    if isinstance(source, network.NetworkSource):
        return f"s16le {network.WIRE_SAMPLE_RATE} Hz mono"
    return None


def _client_configs_of(source) -> list:
    """Per-client configuration lines sent by a network sender.

    ``client_configs`` is a property returning a list, not a method. An
    earlier ``callable()`` guard here therefore always failed and every stored
    event recorded an empty configuration, even though the phone had sent one.
    """
    configs = getattr(source, "client_configs", None)
    if configs is None:
        return []
    if callable(configs):  # tolerate a method-style implementation
        configs = configs()
    return [dict(c) for c in configs]
