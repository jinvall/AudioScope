"""The detection stage.

Consumes analysis results and turns them into stored, classified events.

Threading follows the same rule as the analysis stage (AGENTS.md section 2.3):
this runs on its own thread, is fed by a non-blocking put, and drops rather than
blocks when it falls behind.  A dropped *analysis frame* costs event fidelity,
not audio fidelity, because the audio has already been recorded by the
continuous recorder; the loss is counted and reported.

Ordering matters and is worth stating: a completed event is only returned to
the caller after its post-roll has been written into the ring buffer, so the
extraction below always has the trailing audio available.  That is the reason
the tracker's hold-for-merge step costs latency instead of costing data.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from ..analysis.worker import AnalysisResult
from ..audio.ringbuffer import RingBuffer
from ..config import AppConfig
from .candidate import CandidateDetector
from .classifier import Classification, Classifier
from .tracker import EventProfile, EventTracker, TrackedEvent
from ..events.event import Event, build_event
from ..events.persistence import EventWriter, PendingEvent
from ..events.store import EventStore


@dataclass
class DetectionStats:
    frames_in: int = 0
    frames_dropped: int = 0
    events_detected: int = 0
    events_stored: int = 0
    events_unwritable: int = 0
    write_errors: list[str] = field(default_factory=list)
    # Detection work is measured per analysis frame, which is what the pipeline
    # actually schedules.
    seconds: float = 0.0

    @property
    def mean_frame_ms(self) -> float:
        if not self.frames_in:
            return 0.0
        return 1000.0 * self.seconds / self.frames_in

    def to_dict(self) -> dict:
        return {
            "frames_in": self.frames_in,
            "frames_dropped": self.frames_dropped,
            "events_detected": self.events_detected,
            "events_stored": self.events_stored,
            "events_unwritable": self.events_unwritable,
            "mean_frame_ms": round(self.mean_frame_ms, 4),
            "is_contiguous": self.frames_dropped == 0,
            "write_errors": list(self.write_errors),
        }


_SENTINEL = object()


class DetectionWorker:
    """Runs candidate detection, tracking, classification and storage."""

    def __init__(
        self,
        config: AppConfig,
        ring: RingBuffer,
        store: Optional[EventStore] = None,
        input_queue_size: int = 512,
        on_event: Optional[Callable[[Event], None]] = None,
        writer: Optional[EventWriter] = None,
    ) -> None:
        self.config = config.validate()
        self.ring = ring
        self.store = store if store is not None else EventStore(
            self.config.output_dir
        )
        self.on_event = on_event
        # Persistence is optional and always off the detection thread: a slow
        # disk must not be able to stall event extraction.
        self.writer = writer
        self.sample_rate = self.config.sample_rate
        self.frame_rate = self.config.analysis_frames_per_second

        self._detector = CandidateDetector(
            self.config.detection, frame_rate=self.frame_rate
        )
        self._tracker = EventTracker(
            self.config.detection,
            sample_rate=self.sample_rate,
            frame_rate=self.frame_rate,
            pre_roll_seconds=self.config.event.pre_roll_seconds,
            post_roll_seconds=self.config.event.post_roll_seconds,
        )
        self._classifier = Classifier()

        self._in: queue.Queue = queue.Queue(maxsize=input_queue_size)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.stats = DetectionStats()
        self.events: list[Event] = []
        self._source_info: dict[str, Any] = {}
        # Most recent noise floor, copied into each event's metadata.
        self._last_floor: Optional[float] = None
        # Events whose audio span is not fully captured yet.  An event detected
        # while the stream is still running has its post-roll in the *future*:
        # the tracker has finished deciding the event, but the audio after it has
        # not been captured. Extracting immediately produced an event whose
        # stored audio was several seconds shorter than its recorded span, while
        # still reporting the pre-roll as complete. Holding the event until
        # capture has passed its end sample is what makes the stored audio match
        # the span docs/AUDIO_PIPELINE.md section 10 describes.
        # (event, profile) pairs: the profile is needed to build the
        # fingerprint, and it is cheaper to carry it than to re-derive it.
        self._awaiting_audio: list[tuple[Event, Any]] = []

    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_source_info(self, **info: Any) -> None:
        """Record provenance for every event, e.g. the network client config."""
        self._source_info = dict(info)
        self._source_provider = info.get("provider")

    def refresh_source_info(self) -> None:
        """Re-read the source's provenance now.

        A network source is wired up before any client has connected, so the
        details captured then say "awaiting-client". Reading them per event
        instead is what makes the stored provenance describe the audio that was
        actually recorded, including the config line the phone sent.
        """
        provider = self._source_provider
        if provider is None:
            return
        try:
            self.set_source_info(**provider())
        except Exception:  # pragma: no cover - provenance is best-effort
            pass

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="detection", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------
    def submit(self, result: AnalysisResult) -> bool:
        """Feed one analysis result.  Never blocks."""
        if not self.running:
            return False
        try:
            self._in.put_nowait(result)
            self.stats.frames_in += 1
            return True
        except queue.Full:
            try:
                self._in.get_nowait()
            except queue.Empty:
                pass
            try:
                self._in.put_nowait(result)
            except queue.Full:
                self.stats.frames_dropped += 1
                return False
            self.stats.frames_dropped += 1
            return False

    # ------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            try:
                item = self._in.get(timeout=0.2)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            if item is _SENTINEL:
                break
            try:
                self._process(item)
            except Exception as exc:
                self.stats.write_errors.append(
                    f"detection: {type(exc).__name__}: {exc}"
                )

    def _process(self, result: AnalysisResult) -> None:
        began = time.perf_counter()
        # Remembered so every event records the floor in force when it was
        # detected, which is what makes the stored metadata interpretable.
        self._last_floor = result.floor.overall_floor_db
        candidate = self._detector.update(
            result.features, result.floor, result.start_sample
        )
        tracked = self._tracker.update(
            candidate, result.features, result.floor
        )
        if tracked is not None:
            self._queue_event(tracked, gap=result.gap)
        else:
            self._drain_ready_events()
        self.stats.seconds += time.perf_counter() - began

    # ------------------------------------------------------------------
    def _queue_event(self, tracked: TrackedEvent, gap: bool = False) -> None:
        """Hold a completed event until its post-roll has been captured."""
        event = self._build(tracked, gap)
        self._awaiting_audio.append((event, tracked.profile))
        self._drain_ready_events()

    def _drain_ready_events(self, force: bool = False) -> list[Event]:
        """Extract and store every held event whose span is now available.

        ``force`` is for end of stream.  An event detected near the end of a
        recording can have a post-roll that extends past the last captured
        sample, and that post-roll is never going to exist - the stream has
        stopped.  Those events are stored anyway, with the missing frames
        recorded, because discarding a real detection because its tail was cut
        off would be a worse lie than a short event with an honest note.
        """
        if not self._awaiting_audio:
            return []
        written = self.ring.total_written
        ready = [
            item
            for item in self._awaiting_audio
            if force
            or int(round(item[0].end_seconds * item[0].sample_rate)) <= written
        ]
        if not ready:
            return []
        self._awaiting_audio = [
            item for item in self._awaiting_audio if item not in ready
        ]
        return [self._store(event, profile) for event, profile in ready]

    def _store(self, event: Event, profile: Any) -> Event:
        event, samples = self.store.extract(event, self.ring)
        if samples is None:
            self.stats.events_unwritable += 1
            self.stats.write_errors.append(
                f"{event.event_id}: no audio in the ring buffer for "
                f"{event.start_seconds:.2f}-{event.end_seconds:.2f}s"
            )
        else:
            before = len(self.store.write_errors)
            self.store.save(event, samples)
            if len(self.store.write_errors) > before:
                self.stats.write_errors.append(self.store.write_errors[-1])
                self.stats.events_unwritable += 1
            else:
                self.stats.events_stored += 1
            if event.postroll_missing_frames:
                self.stats.write_errors.append(
                    f"{event.event_id}: {event.postroll_missing_frames} frame(s) "
                    "of post-roll were not captured before the stream ended"
                )
        self.events.append(event)
        self._queue_persistence(event, profile)
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception as exc:
                self.stats.write_errors.append(
                    f"on_event: {type(exc).__name__}: {exc}"
                )
        return event

    def _queue_persistence(self, event: Event, profile: Any) -> None:
        """Hand the event to the background writer.  Never blocks.

        Only the index row goes to the database; the audio and the full
        detector metadata have already been written to the event directory by
        the file store, so dropping this row would only lose the index entry.
        """
        if self.writer is None:
            return
        metadata = dict(event.metadata())
        # Retain the measurements the fingerprint does not carry, so a reviewer
        # can ask why two events were considered similar without re-analysing
        # the audio.
        metadata["band_snr"] = _band_snr_map(profile)
        pending = PendingEvent(
            event_id=event.event_id,
            timestamp=event.detected_at,
            duration=event.duration,
            metadata=metadata,
            profile=profile,
            audio_path=event.audio_path,
            start_seconds=event.start_seconds,
            end_seconds=event.end_seconds,
            classification=event.classification,
        )
        try:
            if not self.writer.submit(pending):
                self.stats.write_errors.append(
                    f"{event.event_id}: persistence queue full; index row "
                    "skipped (audio and metadata are still on disk)"
                )
        except Exception as exc:
            self.stats.write_errors.append(
                f"{event.event_id}: persistence submit failed: "
                f"{type(exc).__name__}: {exc}"
            )

    def _build(self, tracked: TrackedEvent, gap: bool = False) -> Event:
        self.refresh_source_info()
        classification: Classification = self._classifier.classify(
            tracked.profile, self._active_duration(tracked)
        )
        event_id, _day = self.store.allocate_id()
        event = build_event(
            tracked,
            classification,
            event_id,
            noise_floor_db=self._tracker_last_floor,
            peak_snr_db=tracked.profile.peak_snr_db,
            source=self._source_info.get("name", ""),
            stream_rate=self._source_info.get("stream_rate"),
            wire_format=self._source_info.get("wire_format"),
            client_configs=self._source_info.get("client_configs", []),
            gap=gap,
        )
        self.stats.events_detected += 1
        return event

    @property
    def _tracker_last_floor(self) -> Optional[float]:
        return self._last_floor

    def _active_duration(self, tracked: TrackedEvent) -> float:
        """The active part of the event, excluding pre-roll and post-roll.

        Duration limits and the "short event" rules must describe the sound, not
        the padding around it, or a 0.2 s footstep would look like a 10 s event.
        """
        profile = tracked.profile
        return (profile.last_active_sample - profile.onset_sample) / self.sample_rate

    # ------------------------------------------------------------------
    def flush(self) -> list[Event]:
        """Finish the stream, emitting any events still in progress.

        Must be called after the analysis worker has stopped, so the ring buffer
        holds the post-roll of the last event.
        """
        emitted: list[Event] = []
        # Drain anything still queued, then finish the state machine.
        while True:
            try:
                item = self._in.get_nowait()
            except queue.Empty:
                break
            if item is _SENTINEL:
                continue
            try:
                self._process(item)
            except Exception as exc:
                self.stats.write_errors.append(
                    f"detection: {type(exc).__name__}: {exc}"
                )
        for tracked in self._tracker.flush():
            self._queue_event(tracked)
        # At end of stream every remaining span is available, because the ring
        # buffer holds everything that was ever written.
        return self._drain_ready_events(force=True)

    def stop(self, timeout: float = 10.0) -> DetectionStats:
        self._stop.set()
        try:
            self._in.put_nowait(_SENTINEL)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                self.stats.write_errors.append(
                    "detection thread did not stop in time"
                )
            self._thread = None
        return self.stats



def _band_snr_map(profile: Any) -> dict:
    """Per-band SNR of the event, kept alongside the fingerprint.

    Small, and it is what lets a reviewer see *which* part of the spectrum an
    event used without re-running the analysis.
    """
    out: dict = {}
    for name in ("low", "mid", "high"):
        value = getattr(profile, f"mean_{name}_band_snr_db", None)
        if value is not None:
            out[name] = round(float(value), 3)
    return out
