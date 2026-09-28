"""Review controller: the GUI's single, testable accessor to the backend.

The widgets in :mod:`app.gui.widgets` hold no database logic and no audio
logic. They call a :class:`ReviewController`, which is plain Python and can be
tested without a display.

This is what keeps the brief's "one event model, one persistence model"
constraint honest: there is exactly one place the GUI reads events and exactly
one place it writes annotations, and both are here.

Expensive work is never done inline. Audio decoding, envelope building and
spectrogram generation return a :class:`Pending` that a worker thread fills in,
so the view can stay responsive and the detector is never involved.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from ..events.database import Decision, EventDatabase, StoredEvent
from .audioview import (
    AudioCorrupt,
    AudioUnavailable,
    Envelope,
    Spectrogram,
    build_envelope,
    build_spectrogram,
    read_event_audio,
)
from .formatting import describe_event, format_decision

#: Where a viewer looks for new events when nothing pushes them.
DEFAULT_POLL_SECONDS = 2.0


@dataclass
class Pending:
    """A task handed to a background thread.

    The view renders whatever is ready and re-renders when the result arrives,
    so nothing blocks and nothing is computed speculatively.
    """

    kind: str                      # "audio" | "envelope" | "spectrogram"
    event_id: str
    done: bool = False
    samples: Optional[np.ndarray] = None
    envelope: Optional[Envelope] = None
    spectrogram: Optional[Spectrogram] = None
    # The SegmentResult of an extraction, once it has one.
    result: Optional[Any] = None
    error: Optional[str] = None
    error_kind: Optional[str] = None  # "missing" | "corrupt"
    token: int = 0                   # guards against out-of-order results

    @property
    def ok(self) -> bool:
        return self.done and self.error is None


@dataclass
class SeparationAttempt:
    """One completed or attempted separation, as the GUI needs to show it.

    Read from the attempt's own ``metadata.json`` rather than recomputed, so
    the panel reports what the model actually did - including the cost, which
    is the number a user needs to decide whether to run another.
    """

    name: str
    directory: str
    query: str
    status: str
    isolated_path: Optional[str] = None
    enhanced_path: Optional[str] = None
    metadata_path: Optional[str] = None
    input_seconds: float = 0.0
    processing_seconds: float = 0.0
    realtime_ratio: Optional[float] = None
    error: Optional[str] = None
    origin: str = "event"

    @property
    def ok(self) -> bool:
        return self.status == "succeeded" and bool(self.isolated_path)

    @property
    def has_enhanced(self) -> bool:
        return bool(self.enhanced_path) and os.path.exists(self.enhanced_path)

    def summary(self) -> str:
        """One line for the attempts list."""
        if not self.ok:
            return f"{self.name}  {self.query}  failed"
        ratio = (
            f"{self.realtime_ratio:.1f}x realtime"
            if self.realtime_ratio
            else "cost not recorded"
        )
        return (
            f"{self.name}  {self.query}  "
            f"{self.input_seconds:.1f}s in {self.processing_seconds:.0f}s  "
            f"({ratio})"
        )


class ReviewController:
    """Holds the database connection and the current selection."""

    def __init__(
        self,
        database: EventDatabase,
        sample_rate: int = 48000,
        config=None,
        events_root: Optional[str] = None,
        capture_dir: Optional[str] = None,
    ) -> None:
        from ..config import AppConfig

        self.db = database
        self.sample_rate = int(sample_rate)
        self.config = config or AppConfig().validate()
        self._token = 0
        self._cache: dict = {}
        self._lock = threading.Lock()
        # Separation state is created on first use, not here, so a session that
        # never separates anything never starts the model.
        self._worker = None
        self._separator = None
        self._separation_listeners: list = []
        # Audio retention accounting belongs to the *capture* process, which is
        # the one that writes audio; the review window is a separate process
        # and does not maintain it.  It is held here only so an operation that
        # removes audio can keep the persisted snapshot honest, and it is
        # normally None in this process.  Reconciliation re-derives the truth
        # from the filesystem, so nothing depends on it being present.
        self._retention = None
        # Two independent roots, and conflating them is a bug: the events tree
        # is wherever the capture pipeline was told to write, while the
        # separation model is installed inside the project.  A run with
        # ``--events /somewhere/else`` must still find the model, and must
        # still write its results beside the event.
        self._project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self._events_root = os.path.abspath(
            events_root or os.path.join(self._project_root, "events")
        )
        # Where the capture process publishes live status.  Passed rather than
        # assumed, for the same reason as the events root: the launcher may
        # have been told to record somewhere else.
        self._capture_dir = os.path.abspath(
            capture_dir or os.path.join(self._project_root, "recordings")
        )

    #: Waveform envelope width.  Fixed, so envelope cost does not depend on
    #: window length and a very long event cannot produce an unbounded array.
    envelope_columns = 1200

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------
    def list_events(
        self,
        decisions=None,
        label: Optional[str] = None,
        limit: Optional[int] = 200,
    ) -> list:
        """Events for the browser, newest first.

        A limit is applied by the database query rather than by slicing a full
        fetch, so a long-running session does not degrade as the store grows.
        """
        return self.db.list_events(decisions=decisions, label=label,
                                  limit=limit)

    def stats(self) -> dict:
        return self.db.stats()

    def change_marker(self) -> tuple:
        """Cheap aggregate so a viewer can notice new events without a scan."""
        return self.db.change_marker()

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def select(self, event_id: Optional[str]):
        """Load one event and its annotation history.

        Returns ``(StoredEvent, description)`` or ``(None, None)``. Selecting
        never recomputes a fingerprint and never re-runs detection: the stored
        record is authoritative.
        """
        if not event_id:
            return None, None
        stored = self.db.get_event(event_id)
        if stored is None:
            return None, None
        return stored, describe_event(stored)

    def similar(self, event_id: str, limit: int = 5) -> list:
        """Closest stored events, if the API makes it cheap.

        Deliberately small and optional: the brief asks that similarity not
        become the product, and it must not interfere with review.
        """
        stored = self.db.get_event(event_id)
        if stored is None or stored.fingerprint is None:
            return []
        hits = self.db.find_similar(stored.fingerprint, limit=limit + 1)
        out = []
        for other_id, distance in hits:
            if other_id == event_id:
                continue
            other = self.db.get_event(other_id)
            out.append((other_id, round(distance, 4), other))
            if len(out) >= limit:
                break
        return out

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------
    def annotate(
        self,
        event_id: str,
        decision=None,
        label: Optional[str] = None,
        confidence: Optional[float] = None,
        notes: Optional[str] = None,
    ):
        """Append an annotation.  Detector measurements are never touched.

        A decision of ``None`` leaves the decision alone, which is what a
        notes-only edit is: the user changed a field, not the verdict.
        """
        if decision is None and label is None and notes is None and \
                confidence is None:
            return None
        payload = {
            "label": label,
            "confidence": confidence,
            "notes": notes,
        }
        if decision is not None:
            payload["decision"] = Decision.parse(decision)
        # An annotation needs a decision; default to keeping the current state
        # rather than inventing one.
        if "decision" not in payload:
            current = self.db.get_event(event_id)
            payload["decision"] = (
                current.decision if current else Decision.UNREVIEWED
            )
        annotation = self.db.annotate(event_id, **payload)
        return annotation

    def clear_label(self, event_id: str):
        """Remove the current user label.

        Labels are append-only history, so "clearing" records a new annotation
        with an empty label rather than rewriting an earlier one. The previous
        label is still in the history.
        """
        stored = self.db.get_event(event_id)
        decision = stored.decision if stored else Decision.UNREVIEWED
        return self.db.annotate(event_id, decision, label="")

    # ------------------------------------------------------------------
    # Source separation
    #
    # The controller is the only place that knows about separation, exactly as
    # it is the only place that knows about the database.  The widgets emit
    # "separate this event with this query over this region" and nothing else.
    # ------------------------------------------------------------------
    def _separation_root(self) -> str:
        return self._events_root

    def separation_worker(self):
        """The separation worker, created on first use.

        Lazy so that a session that never separates anything never starts the
        model process, and so a missing model degrades to a message rather
        than an exception during construction.
        """
        if getattr(self, "_worker", None) is None:
            from ..separation.separator import build_separator
            from ..separation.worker import SeparationWorker

            self._separator = build_separator(
                self.config.separation, self._project_root
            )
            self._worker = SeparationWorker(
                self._separator,
                config=self.config.separation,
                on_result=self._on_separation_result,
                on_state=self._on_separation_state,
                project_root=self._project_root,
            )
            self._worker.start()
        return self._worker

    def separation_available(self) -> bool:
        reason = self.separation_unavailable_reason()
        return reason is None

    def separation_unavailable_reason(self) -> Optional[str]:
        """Why separation cannot run, or ``None``.  Never raises."""
        try:
            worker = self.separation_worker()
        except Exception as exc:  # pragma: no cover - defensive
            return f"{type(exc).__name__}: {exc}"
        status = worker.status()
        return status.get("unavailable_reason")

    def _on_separation_result(self, result) -> None:
        """Called on the worker thread when a job finishes."""
        for callback in list(self._separation_listeners):
            try:
                callback(result)
            except Exception:
                pass

    def _on_separation_state(self) -> None:
        for callback in list(self._separation_listeners):
            try:
                callback(None)
            except Exception:
                pass

    def add_separation_listener(self, callback: Callable) -> None:
        """Register ``callback(result_or_None)``, called off the GUI thread."""
        self._separation_listeners.append(callback)

    def remove_separation_listener(self, callback: Callable) -> None:
        if callback in self._separation_listeners:
            self._separation_listeners.remove(callback)

    def separate(
        self,
        stored: StoredEvent,
        query: str,
        start_seconds: Optional[float] = None,
        duration_seconds: Optional[float] = None,
    ) -> tuple:
        """Queue a separation.  Returns ``(accepted, message)``.

        Never blocks: separation costs several times realtime, so the job goes
        to the worker and the caller is told only whether it was accepted.
        """
        if not stored.audio_available or not stored.audio_path:
            return False, (
                "this event has no audio to separate; its fingerprint and "
                "measurements are still available"
            )
        if not query or not query.strip():
            return False, "enter a query first"

        from ..separation.jobs import JobOrigin, SeparationJob

        job = SeparationJob(
            event_id=stored.event_id,
            source_path=stored.audio_path,
            query=query.strip(),
            # Empty: the separator places the attempt beside the event's own
            # audio, so repeated attempts never overwrite one another.
            output_directory="",
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            origin=(
                JobOrigin.MANUAL_REGION
                if (start_seconds is not None or duration_seconds is not None)
                else JobOrigin.EVENT
            ),
        )
        worker = self.separation_worker()
        if worker.submit(job):
            return True, f"separating {query.strip()!r} in the background"
        return False, worker.stats.last_error or "the separation queue is full"

    def separation_attempts(self, stored: StoredEvent) -> list:
        """Every separation attempt recorded for an event, oldest first.

        Read from disk rather than kept in memory, so attempts made from the
        command line appear here too.  A missing directory is not an error: it
        just means nothing has been separated yet.
        """
        from ..separation.store import SeparationStore

        directory = self._event_directory(stored)
        if not directory:
            return []
        store = SeparationStore(self._separation_root())
        attempts = []
        for name in store.list_attempts(directory):
            path = os.path.join(directory, name)
            metadata = store.read_metadata(path)
            if not metadata:
                # The attempt directory is created when the job starts and gets
                # its metadata when it finishes, so a directory with no
                # metadata is a job *in progress*.  Listing it as an attempt
                # showed a row with no query and no status, which reads as a
                # failure rather than as work happening.
                continue
            attempts.append(
                SeparationAttempt(
                    name=name,
                    directory=path,
                    query=str(metadata.get("query") or ""),
                    status=str(metadata.get("status") or "unknown"),
                    isolated_path=_existing(metadata.get("isolated_path")),
                    enhanced_path=_existing(metadata.get("enhanced_path")),
                    metadata_path=os.path.join(path, "metadata.json"),
                    input_seconds=float(metadata.get("input_seconds") or 0.0),
                    processing_seconds=float(
                        metadata.get("processing_seconds") or 0.0
                    ),
                    realtime_ratio=metadata.get("realtime_ratio"),
                    error=metadata.get("error"),
                    origin=str(metadata.get("origin") or "event"),
                )
            )
        return attempts

    def recent_queries(self, limit: int = 12) -> list:
        """Queries the user has already run, most recent first.

        Offered as suggestions rather than a fixed list of sound classes,
        because the interface stays application-agnostic: the backend does not
        know what sounds exist, and neither should the window.  What the user
        has already asked for is the honest suggestion list.
        """
        queries: list[str] = []
        seen = set()
        for event in self.db.list_events(limit=60):
            for attempt in self.separation_attempts(event):
                query = attempt.query.strip()
                if query and query not in seen:
                    seen.add(query)
                    queries.append(query)
        return queries[:limit]

    def _event_directory(self, stored: StoredEvent) -> Optional[str]:
        if stored.audio_path and os.path.exists(stored.audio_path):
            return os.path.dirname(stored.audio_path)
        from ..separation.store import SeparationStore

        store = SeparationStore(self._separation_root())
        day = store.find_day(stored.event_id)
        if day is None:
            return None
        return store.event_directory(day, stored.event_id)

    def separation_status(self) -> dict:
        try:
            return self.separation_worker().status()
        except Exception as exc:  # pragma: no cover - defensive
            return {"worker": {}, "model": {}, "available": False,
                    "unavailable_reason": f"{type(exc).__name__}: {exc}"}

    def begin_separation_audio(self, path: Optional[str]) -> Pending:
        """Queue a decode of a separated file, for A/B playback.

        Same background-decoding path as an event's own audio, so comparing the
        original with a separation does not block the window.
        """
        pending = Pending(kind="separation_audio", event_id=path or "",
                          token=self._next_token())
        cache_key = ("separation", path)

        def work() -> None:
            try:
                if not path or not os.path.exists(path):
                    raise AudioUnavailable(f"no separated audio at {path}")
                samples = read_event_audio(path, self.sample_rate)
                pending.samples = samples
                with self._lock:
                    self._cache[cache_key] = samples
            except Exception as exc:
                pending.error = str(exc)
                pending.error_kind = "missing"
            finally:
                pending.done = True

        return _dispatch(work, pending)

    def close_separation(self) -> None:
        """Stop the worker and the model process.  Safe to call twice."""
        worker = getattr(self, "_worker", None)
        if worker is not None:
            worker.stop()
            self._worker = None

    # ------------------------------------------------------------------
    # Clearing unreviewed audio
    # ------------------------------------------------------------------
    def preview_clear_unreviewed(self) -> Any:
        """What clearing unreviewed audio would do.  Touches nothing."""
        from ..events.clear import clear_unreviewed_audio

        return clear_unreviewed_audio(self.db, self._retention, dry_run=True)

    def begin_clear_unreviewed_audio(self) -> Pending:
        """Clear the audio of every unreviewed event, off the GUI thread.

        The preview and the run are the same code with ``dry_run`` flipped, so
        the figures the user confirms are the figures it reports.
        """
        pending = Pending(kind="clear_unreviewed", event_id="", token=self._next_token())

        def work() -> None:
            try:
                from ..events.clear import clear_unreviewed_audio

                pending.result = clear_unreviewed_audio(
                    self.db, self._retention, dry_run=False
                )
            except Exception as exc:
                pending.error = f"{type(exc).__name__}: {exc}"
            finally:
                pending.done = True

        return _dispatch(work, pending)

    def begin_extract_selection(
        self, stored: StoredEvent, start_seconds: float, end_seconds: float
    ) -> Pending:
        """Queue the extraction of a selected region as its own event.

        Off the GUI thread, and for the same reason as every other piece of
        work here: the selection is re-analysed by the real pipeline, which
        costs real CPU, and a review window that freezes is unusable.
        """
        pending = Pending(
            kind="extract_selection", event_id=stored.event_id,
            token=self._next_token(),
        )

        def work() -> None:
            try:
                from ..events.segment import extract_selection

                result = extract_selection(
                    self.config,
                    parent_event_id=stored.event_id,
                    parent_audio_path=stored.audio_path,
                    start_seconds=float(start_seconds),
                    end_seconds=float(end_seconds),
                    event_root=self._events_root,
                    database_path=self.db.path,
                    on_progress=lambda text: setattr(
                        pending, "error_kind", text
                    ),
                )
                pending.result = result
                if not result.ok:
                    pending.error = result.error or "nothing was stored"
            except Exception as exc:
                pending.error = f"{type(exc).__name__}: {exc}"
            finally:
                pending.done = True

        return _dispatch(work, pending)

    def read_live_status(self) -> Optional[dict]:
        """The live capture document, or ``None`` if capture is not publishing.

        Read on every poll rather than cached: this is the one thing in the
        window that must never be stale, because its whole purpose is to say
        whether audio is arriving *now*.
        """
        from ..livestatus import LIVE_STATUS_FILENAME, read_live_status

        return read_live_status(
            os.path.join(self._capture_dir, LIVE_STATUS_FILENAME)
        )

    def capture_dir(self) -> str:
        """Where the capture process publishes its live status."""
        return self._capture_dir

    # ------------------------------------------------------------------
    # Audio and visualisation - always off the GUI thread
    # ------------------------------------------------------------------
    def _next_token(self) -> int:
        self._token += 1
        return self._token

    def begin_audio(self, stored: StoredEvent) -> Pending:
        """Queue a decode of the event's audio.  Returns immediately."""
        pending = Pending(kind="audio", event_id=stored.event_id,
                          token=self._next_token())
        cache_key = ("audio", stored.event_id)

        def work() -> None:
            try:
                if not getattr(stored, "audio_path", None):
                    # Retention may have evicted the audio while the
                    # fingerprint stayed.  Saying so is more useful than a
                    # missing-file error, and it is the truth: the event is
                    # still comparable and still reviewable.
                    evicted = getattr(stored, "audio_evicted_at", None)
                    raise AudioUnavailable(
                        "audio was evicted by retention; the fingerprint and "
                        "measurements are still available"
                        if evicted
                        else "no audio was recorded for this event"
                    )
                samples = read_event_audio(stored.audio_path, self.sample_rate)
                pending.samples = samples
                with self._lock:
                    self._cache[cache_key] = samples
            except AudioUnavailable as exc:
                pending.error = str(exc)
                pending.error_kind = (
                    "corrupt" if isinstance(exc, AudioCorrupt) else "missing"
                )
            except Exception as exc:  # pragma: no cover - defensive
                pending.error = f"{type(exc).__name__}: {exc}"
                pending.error_kind = "corrupt"
            finally:
                pending.done = True

        return _dispatch(work, pending)

    def cached_audio(self, event_id: str) -> Optional[np.ndarray]:
        with self._lock:
            return self._cache.get(("audio", event_id))

    def begin_envelope(self, stored: StoredEvent) -> Pending:
        """Queue a min/max envelope.  Returns immediately."""
        samples = self.cached_audio(stored.event_id)
        if samples is None:
            pending = Pending(kind="envelope", event_id=stored.event_id,
                              token=self._next_token())
            pending.error = "audio not loaded"
            pending.error_kind = "missing"
            pending.done = True
            return pending

        pending = Pending(kind="envelope", event_id=stored.event_id,
                          token=self._next_token())
        columns = self.envelope_columns
        cache_key = ("envelope", stored.event_id, columns)

        def work() -> None:
            try:
                pending.envelope = build_envelope(
                    samples, self.sample_rate, columns=columns
                )
                with self._lock:
                    self._cache[cache_key] = pending.envelope
            except Exception as exc:  # pragma: no cover - defensive
                pending.error = f"{type(exc).__name__}: {exc}"
            finally:
                pending.done = True

        return _dispatch(work, pending)

    def cached_envelope(self, event_id: str,
                        columns: int = 1200) -> Optional[Envelope]:
        with self._lock:
            return self._cache.get(("envelope", event_id, columns))

    def begin_spectrogram(self, stored: StoredEvent) -> Pending:
        """Queue an event-scoped spectrogram.  Returns immediately.

        Never run speculatively: the caller asks for it when the user opens the
        spectrogram, and the result is cached per event.
        """
        samples = self.cached_audio(stored.event_id)
        if samples is None:
            pending = Pending(kind="spectrogram", event_id=stored.event_id,
                              token=self._next_token())
            pending.error = "audio not loaded"
            pending.error_kind = "missing"
            pending.done = True
            return pending

        pending = Pending(kind="spectrogram", event_id=stored.event_id,
                          token=self._next_token())
        cache_key = ("spectrogram", stored.event_id)

        def work() -> None:
            try:
                pending.spectrogram = build_spectrogram(
                    samples, self.sample_rate, config=self.config
                )
                with self._lock:
                    self._cache[cache_key] = pending.spectrogram
            except Exception as exc:  # pragma: no cover - defensive
                pending.error = f"{type(exc).__name__}: {exc}"
            finally:
                pending.done = True

        return _dispatch(work, pending)

    def cached_spectrogram(self, event_id: str) -> Optional[Spectrogram]:
        with self._lock:
            return self._cache.get(("spectrogram", event_id))

    def evict(self, event_id: str) -> None:
        """Drop cached audio and visualisations for one event."""
        with self._lock:
            for key in [k for k in self._cache
                        if k[1] == event_id]:
                self._cache.pop(key, None)

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def cache_size(self) -> int:
        with self._lock:
            return len(self._cache)


# ----------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------
def _dispatch(work: Callable[[], None], pending: Pending) -> Pending:
    """Run ``work`` on a throwaway thread and return the pending at once.

    A throwaway thread per task is deliberate.  A pool would be marginally
    cheaper, but a long waveform decode would then occupy a pool slot and
    delay a later one; with a thread each, the GUI thread never waits and
    tasks cannot queue up behind each other.
    """
    thread = threading.Thread(
        target=work, name=f"gui-{pending.kind}-{pending.event_id}", daemon=True
    )
    thread.start()
    return pending


def _existing(path: Optional[str]) -> Optional[str]:
    """Return the path only if it is really there.

    A separation recorded minutes ago may have been removed since - or its
    audio evicted by retention - and offering a play button for a file that is
    gone is worse than saying there is nothing to play.
    """
    return path if path and os.path.exists(path) else None


def open_controller(
    db_path: str = "events.db",
    config=None,
    events_root: Optional[str] = None,
    capture_dir: Optional[str] = None,
) -> ReviewController:
    """Open the database and build a controller.  Raises if the file is absent.

    ``events_root`` is where the capture pipeline wrote the event directories.
    It defaults to the project's own ``events`` directory, which is right for
    a default run and wrong for every run that passed ``--events``.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"no event database at {os.path.abspath(db_path)}; run a capture "
            f"first, or pass --db"
        )
    database = EventDatabase(db_path)
    from ..config import AppConfig

    config = config or AppConfig().validate()
    return ReviewController(database, sample_rate=config.sample_rate,
                            config=config, events_root=events_root,
                            capture_dir=capture_dir)
