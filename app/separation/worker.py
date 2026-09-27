"""The asynchronous separation worker.

docs/SOURCE_SEPARATION.md section 7: separation is asynchronous, because it
costs about 5x realtime.  Nothing may wait on it - not the GUI, not live
capture, not event detection.  The separation of one event is a background
task whose result appears when it exists.

Structure, and why:

* :meth:`SeparationWorker.submit` does a non-blocking ``put`` on a bounded
  queue and returns immediately, so the GUI thread never stalls on a model.
* One worker thread drains it.  ``SeparationConfig.workers`` is 1 and
  deliberately stays 1: the model process is the bottleneck, so more threads
  would only add contention.
* The model is unloaded after ``model_idle_unload_seconds``.  Keeping it
  resident saves 35 s per job, but holding 4.5 GB forever is not acceptable
  on a 15 GB machine that also runs the capture pipeline and a GUI.
* Idle unloading is checked between jobs only.  A running separation is never
  interrupted - it is the result the user is waiting for.

The worker publishes results through a callback rather than a shared queue,
because the consumer is a GUI that needs a finished event to enable a button,
not a poll loop.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..config import SeparationConfig
from .jobs import JobStatus, SeparationJob, SeparationResult
from .separator import Separator, build_separator


@dataclass
class WorkerStats:
    """Observable progress for the GUI's queue panel."""

    submitted: int = 0
    succeeded: int = 0
    failed: int = 0
    rejected_queue_full: int = 0
    completed: int = 0
    models_loaded: int = 0
    models_unloaded: int = 0
    last_realtime_ratio: Optional[float] = None
    total_processing_seconds: float = 0.0
    last_error: Optional[str] = None
    busy: bool = False
    current_query: Optional[str] = None

    @property
    def pending(self) -> int:
        return max(0, self.submitted - self.completed - self.rejected_queue_full)

    def to_dict(self) -> dict[str, Any]:
        return {
            "submitted": self.submitted,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "rejected_queue_full": self.rejected_queue_full,
            "pending": self.pending,
            "completed": self.completed,
            "models_loaded": self.models_loaded,
            "models_unloaded": self.models_unloaded,
            "last_realtime_ratio": (
                round(self.last_realtime_ratio, 3)
                if self.last_realtime_ratio is not None
                else None
            ),
            "mean_realtime_ratio": (
                round(
                    self.total_processing_seconds
                    / max(1, self.succeeded),
                    3,
                )
                if self.succeeded
                else None
            ),
            "last_error": self.last_error,
            "busy": self.busy,
            "current_query": self.current_query,
        }


class SeparationWorker:
    """Runs separation jobs off the caller's thread."""

    def __init__(
        self,
        separator: Optional[Separator] = None,
        config: Optional[SeparationConfig] = None,
        on_result: Optional[Callable[[SeparationResult], None]] = None,
        on_state: Optional[Callable[[], None]] = None,
        project_root: Optional[str] = None,
    ) -> None:
        self.config = config or SeparationConfig()
        self.separator = separator or build_separator(
            self.config, project_root
        )
        self.on_result = on_result
        self.on_state = on_state
        self.stats = WorkerStats()
        self._queue: "queue.Queue[Optional[SeparationJob]]" = queue.Queue(
            maxsize=self.config.queue_max_jobs
        )
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self._model = getattr(self.separator, "model", None)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="separation-worker",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stopping.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None
        close = getattr(self.separator, "close", None)
        if callable(close):
            close()

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until the queue is empty.  For tests and for shutdown."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._queue.empty() and not self.stats.busy:
                return True
            time.sleep(0.02)
        return False

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------
    def submit(self, job: SeparationJob) -> bool:
        """Queue a job.  Returns False if the queue was full.

        Never blocks and never raises: a full queue means the user has asked
        for more separations than the machine can justify, and the right
        answer is to say so and count it, not to stall the GUI.
        """
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self.stats.rejected_queue_full += 1
            self.stats.last_error = (
                f"separation queue is full ({self.config.queue_max_jobs} "
                f"jobs); wait for the current one to finish"
            )
            self._notify_state()
            return False
        self.stats.submitted += 1
        self._notify_state()
        return True

    def preload_model(self) -> None:
        """Load the model ahead of the first job, on this worker.

        Worth doing while the user is still reading an event: it converts a
        35 s surprise into a 35 s wait at a moment of their choosing.
        """
        self.start()
        self._queue.put(("_preload", None))  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                self._maybe_unload_model()
                continue
            if item is None:
                break
            if isinstance(item, tuple) and item and item[0] == "_preload":
                self._preload()
                continue
            job: SeparationJob = item  # type: ignore[assignment]
            self._execute(job)
        # The model process is stopped with the worker, not left behind.
        close = getattr(self.separator, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _preload(self) -> None:
        if self._model is None or self._model.status.loaded:
            return
        try:
            started = time.time()
            self._model.load()
            self._model.status.load_seconds = time.time() - started
            self.stats.models_loaded += 1
        except Exception as exc:
            self.stats.last_error = f"model preload failed: {exc}"
        self._notify_state()

    def _execute(self, job: SeparationJob) -> None:
        self.stats.busy = True
        self.stats.current_query = job.query
        self._notify_state()
        try:
            result = self.separator.separate(job)
        except Exception as exc:  # noqa: BLE001 - a worker must not die
            result = SeparationResult(
                job=job,
                status=JobStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            self.stats.busy = False
            self.stats.current_query = None

        if result.ok:
            self.stats.succeeded += 1
            self.stats.total_processing_seconds += result.processing_seconds
            if result.realtime_ratio is not None:
                self.stats.last_realtime_ratio = result.realtime_ratio
            if self._model is not None and self._model.status.loaded:
                self.stats.models_loaded = max(1, self.stats.models_loaded)
        else:
            self.stats.failed += 1
            self.stats.last_error = result.error
        self.stats.completed += 1
        if self._model is not None:
            self._model.mark_used()
        self._notify_state()
        if self.on_result is not None:
            try:
                self.on_result(result)
            except Exception:
                # A misbehaving listener must not take the worker down.
                pass
        self._maybe_unload_model()

    def _maybe_unload_model(self) -> None:
        """Release the model when it has been idle long enough.

        Only ever called between jobs, so this can never interrupt a
        separation that is producing the result the user is waiting for.
        """
        if self._model is None:
            return
        if not self._model.idle_expiry_due():
            return
        try:
            self._model.stop()
            self.stats.models_unloaded += 1
        except Exception:
            pass

    def _notify_state(self) -> None:
        if self.on_state is not None:
            try:
                self.on_state()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Combined worker and model status, for the GUI panel."""
        payload: dict[str, Any] = {"worker": self.stats.to_dict()}
        if self._model is not None:
            payload["model"] = self._model.status.to_dict()
        else:
            payload["model"] = {"running": False, "loaded": False}
        reason = None
        getter = getattr(self.separator, "unavailable_reason", None)
        if callable(getter):
            reason = getter()
        payload["available"] = reason is None
        payload["unavailable_reason"] = reason
        return payload
