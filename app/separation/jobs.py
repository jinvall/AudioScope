"""Jobs and results for source separation.

A job is a request to isolate one query from one region of one recording.
It is deliberately a plain value object with no thread and no file handle, so
it can be created on the GUI thread, queued, serialised to metadata, and
reconstructed in a test.

The fields required by docs/SOURCE_SEPARATION.md section 7 are all here;
``origin`` and ``region`` are additions that let the GUI and the metadata
distinguish an automatic event from a manually selected region without
guessing from the path.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"


class JobOrigin(str, Enum):
    """Where the audio being separated came from.

    docs/SOURCE_SEPARATION.md section 13 requires manual selection to enter
    the *same* infrastructure as automatic events, and this is how that is
    recorded rather than inferred.
    """

    EVENT = "event"
    MANUAL_REGION = "manual_region"


@dataclass
class SeparationJob:
    """One requested separation."""

    event_id: str
    source_path: str
    query: str
    output_directory: str
    # An explicit output file, used by the CLI's --output.  When set, the
    # result is written exactly there instead of into a fresh
    # separation_NNN directory, and the metadata sits beside it.  The
    # GUI and the event path never set this, so repeated attempts on one
    # event always get their own directory.
    output_path: Optional[str] = None
    requested_time: float = field(default_factory=time.time)
    model_name: str = "AudioSep"
    # Optional region within the source, in seconds.  A manually selected
    # waveform region is expressed this way rather than by first cutting a
    # temporary file, so the job always points at evidence that still exists.
    start_seconds: Optional[float] = None
    duration_seconds: Optional[float] = None
    origin: JobOrigin = JobOrigin.EVENT
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self) -> None:
        if not self.query or not self.query.strip():
            raise ValueError("query must not be empty")
        if self.start_seconds is not None and self.start_seconds < 0:
            raise ValueError("start_seconds must not be negative")
        if self.duration_seconds is not None and self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")

    @property
    def is_region(self) -> bool:
        return self.start_seconds is not None or self.duration_seconds is not None


@dataclass
class SeparationResult:
    """The outcome of one job, successful or not.

    On failure ``error`` is populated and the original event is untouched -
    a failed separation never destroys anything (docs/SOURCE_SEPARATION.md 14).
    """

    job: SeparationJob
    status: JobStatus = JobStatus.QUEUED
    isolated_path: Optional[str] = None
    enhanced_path: Optional[str] = None
    metadata_path: Optional[str] = None
    error: Optional[str] = None

    # Processing metrics (docs/SOURCE_SEPARATION.md section 15).
    input_seconds: float = 0.0
    processing_seconds: float = 0.0
    realtime_ratio: Optional[float] = None
    model_load_seconds: float = 0.0
    model_rss_mb: float = 0.0
    enhancement: dict[str, float] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    attempt: int = 1

    @property
    def ok(self) -> bool:
        return self.status is JobStatus.SUCCEEDED and bool(self.isolated_path)

    @property
    def duration_seconds(self) -> float:
        return self.processing_seconds - self.input_seconds

    def to_metadata(self) -> dict[str, Any]:
        """The record written to ``metadata.json``.

        Only measured values appear.  ``confidence``-style fields are absent
        entirely rather than filled with something plausible, matching the
        project's convention that a missing measurement reads as absent.
        """
        job = self.job
        return {
            "job_id": job.job_id,
            "event_id": job.event_id,
            "origin": job.origin.value,
            "query": job.query,
            "model": job.model_name,
            "source_path": job.source_path,
            "start_seconds": job.start_seconds,
            "duration_seconds": job.duration_seconds,
            "attempt": self.attempt,
            "output_path": job.output_path,
            "requested_time": job.requested_time,
            "status": self.status.value,
            "isolated_path": self.isolated_path,
            "enhanced_path": self.enhanced_path,
            "error": self.error,
            "input_seconds": round(self.input_seconds, 4),
            "processing_seconds": round(self.processing_seconds, 4),
            "realtime_ratio": (
                round(self.realtime_ratio, 3)
                if self.realtime_ratio is not None
                else None
            ),
            "model_load_seconds": round(self.model_load_seconds, 3),
            "model_rss_mb": round(self.model_rss_mb, 1),
            "enhancement": self.enhancement,
            "validation": self.validation,
            "warnings": list(self.warnings),
        }
