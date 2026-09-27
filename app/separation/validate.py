"""Output validation for a separation result.

docs/SOURCE_SEPARATION.md section 9: a job is not successful merely because
the model returned.  Each check below corresponds to a way a returned file
can be unusable, and each failure is reported as evidence rather than
collapsed into a boolean.

The distinction that matters: **a quiet result is not a failed result.**  If
the query names something genuinely absent, the correct output is a very
low-level file, and that is a valid, playable, informative separation.  What
is rejected is output that is empty, unreadable, non-finite, or clipped -
the failures that indicate a broken run rather than an absent sound.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..audio.wavio import read_wav


class SeparationInvalid(RuntimeError):
    """The model's output file failed validation and must not be published."""


@dataclass
class ValidationReport:
    """What was measured about the output, and whether it is usable."""

    path: str
    ok: bool = True
    sample_rate: int = 0
    frames: int = 0
    duration: float = 0.0
    peak: float = 0.0
    peak_dbfs: float = 0.0
    rms: float = 0.0
    rms_dbfs: float = 0.0
    non_finite_samples: int = 0
    clipped_fraction: float = 0.0
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "ok": self.ok,
            "sample_rate": self.sample_rate,
            "frames": self.frames,
            "duration": round(self.duration, 6),
            "peak": round(self.peak, 6),
            "peak_dbfs": round(self.peak_dbfs, 2),
            "rms": round(self.rms, 8),
            "rms_dbfs": round(self.rms_dbfs, 2),
            "non_finite_samples": self.non_finite_samples,
            "clipped_fraction": round(self.clipped_fraction, 6),
            "problems": list(self.problems),
        }


def _dbfs(value: float) -> float:
    """Amplitude ratio to dBFS.  Zero is reported as the measurement floor.

    ``-inf`` is technically correct for digital silence but unusable in a
    metadata field, and the project's convention is that a missing
    measurement reads as absent rather than as a number that looks extreme.
    """
    if value <= 0.0:
        return -200.0
    return float(20.0 * np.log10(value))


def validate_output(
    path: str,
    expected_sample_rate: Optional[int] = None,
    min_peak_dbfs: float = -80.0,
    max_clipped_fraction: float = 0.001,
    expected_duration: Optional[float] = None,
    duration_tolerance: float = 0.25,
) -> ValidationReport:
    """Inspect a separated file and decide whether it may be published.

    ``expected_duration`` is checked because a truncated file is otherwise
    indistinguishable from a short one: the model returning fewer samples
    than the input produced is a failure mode worth catching.
    """
    report = ValidationReport(path=path)

    if not os.path.exists(path):
        report.ok = False
        report.problems.append("output file was not created")
        return report

    size = os.path.getsize(path)
    if size == 0:
        report.ok = False
        report.problems.append("output file is empty")
        return report

    try:
        audio = read_wav(path)
    except Exception as exc:
        report.ok = False
        report.problems.append(f"output file cannot be read: {exc}")
        return report

    report.sample_rate = _file_sample_rate(path)
    report.frames = int(audio.size)

    if audio.size == 0:
        report.ok = False
        report.problems.append("output contains no samples")
        return report
    if not np.isfinite(audio).all():
        report.non_finite_samples = int((~np.isfinite(audio)).sum())
        report.ok = False
        report.problems.append(
            f"output contains {report.non_finite_samples} non-finite samples "
            "(NaN or Inf)"
        )
        return report

    report.duration = audio.size / float(report.sample_rate)
    if report.duration <= 0.0:
        report.ok = False
        report.problems.append("output duration is zero")
        return report

    finite = audio[np.isfinite(audio)]
    report.peak = float(np.max(np.abs(finite))) if finite.size else 0.0
    report.rms = float(np.sqrt(np.mean(finite ** 2))) if finite.size else 0.0
    report.peak_dbfs = _dbfs(report.peak)
    report.rms_dbfs = _dbfs(report.rms)
    report.clipped_fraction = float(
        np.mean(np.abs(finite) >= 0.999) if finite.size else 0.0
    )

    if expected_sample_rate and report.sample_rate != expected_sample_rate:
        report.ok = False
        report.problems.append(
            f"sample rate is {report.sample_rate}, expected "
            f"{expected_sample_rate}"
        )
    if report.peak_dbfs < min_peak_dbfs:
        # Not a failure.  Recorded so the UI can say "the model found little
        # of that here" instead of implying a broken run.
        report.problems.append(
            f"output peak is {report.peak_dbfs:.1f} dBFS, below "
            f"{min_peak_dbfs:.1f} dBFS: the query may not match this audio"
        )
    if report.clipped_fraction > max_clipped_fraction:
        report.ok = False
        report.problems.append(
            f"output is clipped: {report.clipped_fraction * 100:.2f}% of "
            f"samples are at full scale"
        )
    if expected_duration is not None and expected_duration > 0:
        drift = abs(report.duration - expected_duration)
        if drift > duration_tolerance:
            report.ok = False
            report.problems.append(
                f"output duration is {report.duration:.3f}s but the input was "
                f"{expected_duration:.3f}s (truncated or padded)"
            )

    return report


def _file_sample_rate(path: str) -> int:
    import soundfile as sf

    try:
        return int(sf.info(path).samplerate)
    except Exception:
        return 0


def require_valid(report: ValidationReport) -> ValidationReport:
    """Raise if the report is unusable, otherwise return it."""
    if not report.ok:
        raise SeparationInvalid(
            "separation output rejected: " + "; ".join(report.problems)
        )
    return report
