"""The application-level source separation interface.

docs/SOURCE_SEPARATION.md section 3 asks for a stable interface around the
real model, and section 3's warning - "do not invent an API" - is why the
model's own inference is called rather than reimplemented.  This module owns
everything *around* the model instead: preflight, region extraction, output
validation, optional enhancement, and metadata.

    separator.available()          # can separation run at all, and why not
    result = separator.separate(job)

Two implementations exist deliberately:

* :class:`AudioSepSeparator` - the real one.
* :class:`UnavailableSeparator` - returns a failed result with the reason.

The second is not a stub.  It exists so that a missing model, a missing venv,
or a wrong configuration produces an honest, reportable failure through the
normal path, instead of an exception at the GUI.  It never writes an audio
file; that is the line AGENTS.md section 2.4 draws.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional, Protocol, runtime_checkable

from ..audio.wavio import read_wav, write_wav_atomic
from ..config import SeparationConfig
from .enhance import enhance
from .jobs import JobStatus, SeparationJob, SeparationResult
from .model import ModelProcess, ModelUnavailable
from .store import SeparationStore
from .validate import validate_output

ISOLATED_NAME = "isolated.wav"
ENHANCED_NAME = "enhanced.wav"


@runtime_checkable
class Separator(Protocol):
    """What the rest of the application is allowed to depend on."""

    def available(self) -> bool:
        ...

    def unavailable_reason(self) -> Optional[str]:
        ...

    def separate(self, job: SeparationJob) -> SeparationResult:
        ...


class AudioSepSeparator:
    """Query-based separation using the AudioSep model."""

    def __init__(
        self,
        config: Optional[SeparationConfig] = None,
        project_root: Optional[str] = None,
        store: Optional[SeparationStore] = None,
        model: Optional[ModelProcess] = None,
    ) -> None:
        self.config = config or SeparationConfig()
        self.project_root = os.path.abspath(project_root or os.getcwd())
        self.store = store or SeparationStore(
            root=os.path.join(self.project_root, "events")
        )
        self.model = model or ModelProcess(self.config, self.project_root)

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------
    def preflight_problems(self) -> list[str]:
        return self.model.preflight()

    def available(self) -> bool:
        return not self.preflight_problems()

    def unavailable_reason(self) -> Optional[str]:
        problems = self.preflight_problems()
        if not problems:
            return None
        return "source separation is not available:\n  - " + "\n  - ".join(
            problems
        )

    # ------------------------------------------------------------------
    # Separation
    # ------------------------------------------------------------------
    def separate(self, job: SeparationJob) -> SeparationResult:
        result = SeparationResult(job=job, status=JobStatus.RUNNING)
        attempt = self._attempt_directory(job, result)
        if not attempt and not result.error:
            result.error = "no output directory could be determined"
        if result.error:
            result.status = JobStatus.FAILED
            return result

        try:
            model_was_loaded = self.model.status.loaded
            source_for_model, expected_duration = self._prepare_input(
                job, attempt, result
            )
            isolated = self._isolated_path(job, attempt)
            if job.output_path:
                parent = os.path.dirname(isolated)
                if parent:
                    os.makedirs(parent, exist_ok=True)
            started = time.time()
            response = self.model.separate(
                input_path=source_for_model,
                query=job.query.strip(),
                output_path=isolated,
            )
            # The protocol carries an explicit success flag, and it has to be
            # checked: a runner that failed returns a response just like one
            # that succeeded, and without this the failure shows up much later
            # as "the output file was not created", which says nothing about
            # why.
            if not response.get("ok"):
                detail = response.get("error") or "no reason given"
                stderr_tail = self.model.stderr_tail(8)
                raise RuntimeError(
                    f"the model process reported a failure: {detail}"
                    + ("\n  " + "\n  ".join(stderr_tail) if stderr_tail else "")
                )
            self.model.mark_used()
            result.processing_seconds = time.time() - started
            result.input_seconds = float(response.get("input_seconds") or 0.0)
            result.realtime_ratio = response.get("realtime_ratio")
            result.model_rss_mb = float(response.get("model_rss_mb") or 0.0)
            if not model_was_loaded:
                # First job after a cold start; the load is part of its cost.
                result.model_load_seconds = round(
                    self.model.status.load_seconds, 3
                )
                result.warnings.append(
                    "first separation after a cold start: the model load is "
                    "included in the reported processing time"
                )

            report = validate_output(
                isolated,
                expected_sample_rate=self.config.model_sample_rate,
                min_peak_dbfs=self.config.min_output_peak_dbfs,
                max_clipped_fraction=self.config.max_clipped_fraction,
                expected_duration=expected_duration,
            )
            result.validation = report.to_dict()
            result.warnings.extend(report.problems)
            if not report.ok:
                # An invalid file is removed rather than published: a
                # half-written or clipped separation must never be offered
                # for playback as if it were a result.
                _remove(isolated)
                result.status = JobStatus.FAILED
                result.error = "; ".join(report.problems)
                self._finalise(attempt, result)
                return result

            result.isolated_path = isolated
            if self.config.enable_enhancement:
                enhanced = self._enhance(
                    isolated, self._enhanced_path(job, attempt), result
                )
                result.enhanced_path = enhanced
            result.status = JobStatus.SUCCEEDED
        except ModelUnavailable as exc:
            result.status = JobStatus.FAILED
            result.error = str(exc)
        except TimeoutError as exc:
            result.status = JobStatus.FAILED
            result.error = f"separation timed out: {exc}"
        except Exception as exc:  # noqa: BLE001 - reported in the result
            result.status = JobStatus.FAILED
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            # The region cut is deliberately *not* deleted.  It is written
            # into the attempt directory, beside the result it produced, and
            # it is the exact audio the model saw: without it a reviewer
            # cannot tell which part of a long recording a separation
            # actually came from, or reproduce the run.
            self._finalise(attempt, result)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _attempt_directory(
        self, job: SeparationJob, result: SeparationResult
    ) -> str:
        """Resolve the directory this attempt writes into.

        Three cases, and the difference matters:

        * ``job.output_path`` set - the caller named an exact file, so the
          attempt is that file's own directory.  No ``separation_NNN`` is
          created, because the caller has already chosen where it goes.
        * ``job.output_directory`` set - the event case: a fresh
          ``separation_NNN`` is allocated inside it, so a second attempt
          never overwrites the first.
        * neither - fall back to the source file's parent, which is where the
          audio it came from already lives.
        """
        if job.output_path:
            return os.path.dirname(os.path.abspath(job.output_path)) or "."

        event_directory = job.output_directory
        if not event_directory:
            # The source file is the event's own ``original.wav``, so its
            # parent *is* the event directory.  Preferring it is what keeps an
            # attempt beside the audio it came from, and it is the only way to
            # be safe when the events tree this process was given is not the
            # one the event actually lives in: a day-directory search by event
            # id can match a different, unrelated event with the same id, and
            # the attempt would then be written into its evidence.
            source_directory = os.path.dirname(
                os.path.abspath(job.source_path)
            )
            if os.path.isdir(source_directory):
                event_directory = source_directory
            else:
                day = self.store.find_day(job.event_id)
                event_directory = (
                    self.store.event_directory(day, job.event_id)
                    if day
                    else os.path.dirname(os.path.abspath(job.source_path))
                )

        try:
            attempt = self.store.allocate_attempt(event_directory)
        except FileExistsError as exc:
            result.error = f"could not create a separation directory: {exc}"
            return ""

        result.attempt = len(self.store.list_attempts(event_directory))
        return attempt

    def _isolated_path(self, job: SeparationJob, attempt: str) -> str:
        if job.output_path:
            return os.path.abspath(job.output_path)
        return os.path.join(attempt, ISOLATED_NAME)

    def _enhanced_path(self, job: SeparationJob, attempt: str) -> str:
        if job.output_path:
            stem, extension = os.path.splitext(
                os.path.basename(os.path.abspath(job.output_path))
            )
            return os.path.join(
                os.path.dirname(os.path.abspath(job.output_path)) or ".",
                f"{stem}_enhanced{extension or '.wav'}",
            )
        return os.path.join(attempt, ENHANCED_NAME)

    def _prepare_input(
        self,
        job: SeparationJob,
        attempt: str,
        result: SeparationResult,
    ) -> tuple[str, float]:
        """Return (path to feed the model, expected duration in seconds).

        A whole event is passed through untouched.  A region is cut into the
        attempt directory as ``region.wav``: cutting to a *new* file is what
        keeps ``original.wav`` intact (AGENTS.md section 2.1), and keeping the
        cut beside its own result means the exact audio the model saw can
        always be re-examined.
        """
        if not job.is_region:
            duration = _duration_of(job.source_path)
            self._check_bounds(duration, result)
            return job.source_path, duration

        samples = read_wav(job.source_path)
        sample_rate = _sample_rate_of(job.source_path)
        total = samples.size / float(sample_rate)
        start = job.start_seconds or 0.0
        duration = (
            job.duration_seconds
            if job.duration_seconds is not None
            else max(0.0, total - start)
        )
        if start >= total:
            raise ValueError(
                f"region starts at {start:.3f}s but the source is only "
                f"{total:.3f}s long"
            )
        begin = int(round(start * sample_rate))
        length = int(round(duration * sample_rate))
        region = samples[begin:begin + length]
        if region.size == 0:
            raise ValueError("selected region contains no samples")

        path = os.path.join(attempt, "region.wav")
        write_wav_atomic(path, region, sample_rate)
        return path, region.size / float(sample_rate)

    def _check_bounds(self, duration: float, result: SeparationResult) -> None:
        if duration <= 0.0:
            raise ValueError("source contains no audio")
        if duration > self.config.max_input_seconds:
            raise ValueError(
                f"source is {duration:.1f}s, over the "
                f"{self.config.max_input_seconds:.0f}s separation limit; "
                f"select a region instead"
            )
        if duration < self.config.min_input_seconds:
            raise ValueError(
                f"source is {duration:.3f}s, under the "
                f"{self.config.min_input_seconds:.2f}s minimum"
            )

    def _enhance(
        self,
        isolated: str,
        path: str,
        result: SeparationResult,
    ) -> Optional[str]:
        """Write the enhanced copy.  Failure here never fails the job.

        It is always a *separate* file: the raw model output is the
        authoritative separation (docs/SOURCE_SEPARATION.md section 11) and
        enhancement must never overwrite it.
        """
        from ..audio.wavio import read_wav as _read

        try:
            samples = _read(isolated)
            enhanced, applied = enhance(
                samples,
                self.config.enhance_target_peak_dbfs,
                self.config.enhance_remove_dc,
            )
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            write_wav_atomic(
                path, enhanced, self.config.model_sample_rate
            )
            result.enhancement = {k: float(v) for k, v in applied.items()}
            return path
        except Exception as exc:  # noqa: BLE001 - optional step
            result.warnings.append(f"enhancement skipped: {exc}")
            return None

    def _finalise(
        self, attempt: str, result: SeparationResult
    ) -> None:
        """Always write metadata, including for a failed job.

        A failure is evidence too: the record of what was asked, what failed
        and why is what stops the same request being retried blindly.

        With an explicit output path the metadata is written beside that file
        under its own name, never as a bare ``metadata.json`` in a directory
        that may hold unrelated files.
        """
        try:
            output_path = result.job.output_path
            if output_path:
                stem = os.path.splitext(
                    os.path.basename(os.path.abspath(output_path))
                )[0]
                directory = os.path.dirname(
                    os.path.abspath(output_path)
                ) or "."
                result.metadata_path = self.store.write_metadata_as(
                    os.path.join(directory, f"{stem}.metadata.json"),
                    result.to_metadata(),
                )
            else:
                result.metadata_path = self.store.write_metadata(
                    attempt, result.to_metadata()
                )
        except OSError:
            result.metadata_path = None

    def close(self) -> None:
        self.model.stop()


class UnavailableSeparator:
    """Reports a real reason instead of producing a real-looking file."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def available(self) -> bool:
        return False

    def unavailable_reason(self) -> Optional[str]:
        return self.reason

    def separate(self, job: SeparationJob) -> SeparationResult:
        return SeparationResult(
            job=job,
            status=JobStatus.FAILED,
            error=self.reason,
        )


def build_separator(
    config: Optional[SeparationConfig] = None,
    project_root: Optional[str] = None,
) -> Separator:
    """Return a working separator, or an honest one that explains itself."""
    config = config or SeparationConfig()
    if not config.enabled:
        return UnavailableSeparator("source separation is disabled")
    separator = AudioSepSeparator(config, project_root)
    reason = separator.unavailable_reason()
    if reason:
        return UnavailableSeparator(reason)
    return separator


# ----------------------------------------------------------------------
def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _info(path: str) -> Any:
    import soundfile as sf

    return sf.info(path)


def _duration_of(path: str) -> float:
    info = _info(path)
    return float(info.frames) / float(info.samplerate)


def _sample_rate_of(path: str) -> int:
    return int(_info(path).samplerate)
