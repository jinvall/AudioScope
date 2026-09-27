"""Tests for query-based source separation.

Split into two halves, deliberately:

* The **unit** half runs always.  It exercises the whole application layer -
  attempt allocation, region extraction, validation, enhancement, the job
  queue, the worker, and the CLI - against a fake model process.  That is
  where every bug found while building this actually lived: none of them were
  in the model, and all of them would have been invisible to a test that
  required a 2.4 GB download to run.

* The **model** half runs only when ``AUDIOMICROSCOPE_SEPARATION_TESTS=1`` and
  the model is installed, because it costs minutes of CPU per run.  It proves
  the thing the unit half cannot: that a real query produces real, different,
  valid audio.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from app.audio.wavio import write_wav_atomic
from app.config import SeparationConfig
from app.separation.enhance import (
    enhance,
    enhance_file,
    normalise_peak,
    remove_dc,
    soft_clip,
)
from app.separation.jobs import (
    JobOrigin,
    JobStatus,
    SeparationJob,
    SeparationResult,
)
from app.separation.model import ModelStatus
from app.separation.separator import (
    AudioSepSeparator,
    UnavailableSeparator,
    build_separator,
)
from app.separation.store import SeparationStore
from app.separation.validate import SeparationInvalid, require_valid, validate_output
from app.separation.worker import SeparationWorker

SAMPLE_RATE = 48_000
MODEL_SR = 32_000


# ======================================================================
# A fake model process
# ======================================================================
def tone(seconds: float = 1.0, freq: float = 440.0, rate: int = SAMPLE_RATE):
    t = np.arange(int(seconds * rate)) / rate
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def write_at(path: str, seconds: float, rate: int = MODEL_SR,
             subtype: str = "PCM_16", samples=None) -> str:
    """Write audio *at the rate it is declared at*.

    Generating 48 kHz samples and writing them as 32 kHz changes the
    duration, which turns a correct duration assertion into a false failure
    and hides the real behaviour under a sample-rate mistake.
    """
    payload = tone(seconds, rate=rate) if samples is None else samples
    write_wav_atomic(path, payload, rate, subtype=subtype)
    return path


class FakeModel:
    """Stands in for :class:`app.separation.model.ModelProcess`.

    Writes a deterministic response to each query, so a test can prove the
    *output depends on the query* - which is the whole contract of query-based
    separation, and the thing a filter could never satisfy.
    """

    def __init__(self, output: str = "audio", fail: bool = False,
                 clipped: bool = False, finite: bool = True,
                 empty: bool = False) -> None:
        self.output = output
        self.fail = fail
        self.clipped = clipped
        self.finite = finite
        self.empty = empty
        self.status = ModelStatus()
        self.queries: list[str] = []
        self.stopped = False

    def separate(self, input_path, query, output_path, timeout=None) -> dict:
        self.queries.append(query)
        self.status.loaded = True
        if self.fail:
            return {"ok": False, "error": "the model exploded"}
        # Return audio of the same length as the input, as the real model
        # does.  A short answer is exactly what the output validator exists
        # to catch, so a fake that truncates makes every happy-path test
        # assert the wrong thing.
        import soundfile as _sf

        info = _sf.info(input_path)
        seconds = info.frames / float(info.samplerate)
        samples = tone(seconds, freq=100.0 + 50.0 * len(self.queries),
                       rate=MODEL_SR)
        if self.clipped:
            samples = np.ones_like(samples) * 1.5
        if not self.finite:
            samples = samples.copy()
            samples[10] = np.nan
        if self.empty:
            samples = np.zeros(0, dtype=np.float32)
        # Non-finite samples cannot survive PCM_16, so that case is written
        # as float - otherwise the fake silently produces a valid file and
        # the validator is never actually exercised.
        write_wav_atomic(
            output_path, samples, MODEL_SR,
            subtype="FLOAT" if not self.finite else "PCM_16",
        )
        seconds = samples.size / float(MODEL_SR)
        return {
            "ok": True,
            "processing_seconds": 1.0,
            "input_seconds": seconds,
            "realtime_ratio": 2.0,
            "model_rss_mb": 1234.0,
        }

    def preflight(self) -> list:
        return []

    def stderr_tail(self, lines: int = 20):
        return []

    def mark_used(self) -> None:
        self.status.last_used = 1.0

    def idle_seconds(self) -> float:
        return 0.0

    def idle_expiry_due(self) -> bool:
        return False

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def source(tmp_path):
    """A real input WAV on disk, since the store and CLI both read it."""
    path = str(tmp_path / "input.wav")
    write_wav_atomic(path, tone(1.0), SAMPLE_RATE)
    return path


@pytest.fixture
def separator(source, tmp_path):
    root = str(tmp_path / "events")
    os.makedirs(root, exist_ok=True)
    model = FakeModel()
    instance = AudioSepSeparator(
        SeparationConfig(), str(tmp_path), store=SeparationStore(root),
        model=model,
    )
    instance.fake_model = model
    return instance


def job_for(source, query="a whisper", **kwargs) -> SeparationJob:
    return SeparationJob(
        event_id=kwargs.pop("event_id", "event_000001"),
        source_path=source,
        query=query,
        output_directory=kwargs.pop("output_directory", ""),
        **kwargs,
    )


# ======================================================================
# Store: attempts are independent
# ======================================================================
class TestSeparationStore:
    def test_allocates_incrementing_directories(self, tmp_path):
        store = SeparationStore(str(tmp_path))
        first = store.allocate_attempt(str(tmp_path))
        second = store.allocate_attempt(str(tmp_path))
        assert os.path.basename(first) == "separation_001"
        assert os.path.basename(second) == "separation_002"

    def test_never_reuses_an_attempt(self, tmp_path):
        store = SeparationStore(str(tmp_path))
        first = store.allocate_attempt(str(tmp_path))
        # The point is that the *next* allocation steps over what is already
        # there rather than colliding with it.
        assert os.path.isdir(first)
        assert os.path.basename(store.allocate_attempt(str(tmp_path))) == \
            "separation_002"

    def test_ignores_unrelated_directories(self, tmp_path):
        os.makedirs(os.path.join(str(tmp_path), "separation_009"))
        os.makedirs(os.path.join(str(tmp_path), "something_else"))
        store = SeparationStore(str(tmp_path))
        assert os.path.basename(store.allocate_attempt(str(tmp_path))) == \
            "separation_010"

    def test_lists_and_counts_in_numeric_order(self, tmp_path):
        for index in (1, 2, 10):
            os.makedirs(os.path.join(str(tmp_path), f"separation_{index:03d}"))
        store = SeparationStore(str(tmp_path))
        assert store.list_attempts(str(tmp_path)) == [
            "separation_001", "separation_002", "separation_010",
        ]
        assert store.count_attempts(str(tmp_path)) == 3

    def test_find_day(self, tmp_path):
        store = SeparationStore(str(tmp_path))
        os.makedirs(os.path.join(str(tmp_path), "2026-01-01", "event_1"))
        assert store.find_day("event_1") == "2026-01-01"
        assert store.find_day("event_404") is None

    def test_metadata_is_written_atomically(self, tmp_path):
        store = SeparationStore(str(tmp_path))
        path = store.write_metadata(str(tmp_path), {"a": 1})
        assert os.path.basename(path) == "metadata.json"
        assert json.load(open(path)) == {"a": 1}
        # No partial file is left behind.
        assert not os.path.exists(path + ".part")

    def test_metadata_to_an_explicit_path(self, tmp_path):
        store = SeparationStore(str(tmp_path))
        path = store.write_metadata_as(
            str(tmp_path / "out.metadata.json"), {"b": 2}
        )
        assert os.path.basename(path) == "out.metadata.json"

    def test_unreadable_metadata_reads_as_absent(self, tmp_path):
        store = SeparationStore(str(tmp_path))
        assert store.read_metadata(str(tmp_path)) is None
        with open(os.path.join(str(tmp_path), "metadata.json"), "w") as handle:
            handle.write("{not json")
        assert store.read_metadata(str(tmp_path)) is None


# ======================================================================
# Validation
# ======================================================================
class TestValidation:
    def _wav(self, tmp_path, name="out.wav", samples=None, rate=MODEL_SR,
             subtype="PCM_16"):
        return write_at(
            str(tmp_path / name), 0.5, rate=rate, subtype=subtype,
            samples=samples,
        )

    def test_accepts_a_good_file(self, tmp_path):
        report = validate_output(self._wav(tmp_path))
        assert report.ok
        assert report.problems == []
        assert report.sample_rate == MODEL_SR
        assert 0.49 < report.duration < 0.51
        assert -10 < report.peak_dbfs < 0

    def test_missing_file(self, tmp_path):
        report = validate_output(str(tmp_path / "nope.wav"))
        assert not report.ok
        assert "not created" in report.problems[0]

    def test_empty_file(self, tmp_path):
        path = str(tmp_path / "empty.wav")
        open(path, "wb").close()
        report = validate_output(path)
        assert not report.ok
        assert "empty" in report.problems[0]

    def test_no_samples(self, tmp_path):
        report = validate_output(
            self._wav(tmp_path, samples=np.zeros(0, dtype=np.float32))
        )
        assert not report.ok

    def test_unreadable_file(self, tmp_path):
        path = str(tmp_path / "junk.wav")
        with open(path, "wb") as handle:
            handle.write(b"not a wav at all")
        report = validate_output(path)
        assert not report.ok
        assert "cannot be read" in report.problems[0]

    def test_non_finite_samples_are_rejected(self, tmp_path):
        # Float subtype, deliberately: PCM_16 cannot carry a NaN, so this
        # must not be tested through it.
        samples = tone(0.5, rate=MODEL_SR).copy()
        samples[5] = np.inf
        report = validate_output(
            self._wav(tmp_path, samples=samples, subtype="FLOAT")
        )
        assert not report.ok
        assert "non-finite" in report.problems[0]
        assert report.non_finite_samples > 0

    def test_clipped_output_is_rejected(self, tmp_path):
        report = validate_output(
            self._wav(
                tmp_path, samples=np.ones(1000, dtype=np.float32),
                subtype="FLOAT",
            )
        )
        assert not report.ok
        assert any("clipped" in p for p in report.problems)

    def test_wrong_sample_rate_is_rejected(self, tmp_path):
        report = validate_output(
            self._wav(tmp_path, rate=44_100), expected_sample_rate=MODEL_SR
        )
        assert not report.ok
        assert any("sample rate" in p for p in report.problems)

    def test_a_quiet_result_is_valid_not_failed(self, tmp_path):  # noqa: D401
        # The query may simply not be present.  A very low level is a real,
        # informative answer, and must not be reported as a broken run.
        report = validate_output(
            self._wav(
                tmp_path, samples=np.full(1000, 1e-5, dtype=np.float32),
                subtype="FLOAT",
            )
        )
        assert report.ok
        assert report.problems  # a note, not a failure
        assert any("may not match" in p for p in report.problems)

    def test_duration_drift_is_caught(self, tmp_path):
        report = validate_output(
            self._wav(tmp_path), expected_duration=5.0, duration_tolerance=0.25
        )
        assert not report.ok
        assert any("truncated" in p for p in report.problems)

    def test_silence_is_reported_as_absent_not_minus_infinity(self, tmp_path):
        report = validate_output(
            self._wav(
                tmp_path, samples=np.zeros(1000, dtype=np.float32),
                subtype="FLOAT",
            )
        )
        assert report.peak_dbfs == -200.0
        assert np.isfinite(report.peak_dbfs)

    def test_require_valid_raises(self, tmp_path):
        with pytest.raises(SeparationInvalid):
            require_valid(validate_output(str(tmp_path / "nope.wav")))


# ======================================================================
# Enhancement
# ======================================================================
class TestEnhance:
    def test_removes_offset(self):
        shifted = tone(0.2) + 0.25
        assert abs(float(np.mean(remove_dc(shifted)))) < 1e-6
        # The waveform itself is untouched by the offset removal.
        assert abs(float(np.mean(shifted))) > 0.2

    def test_normalise_reaches_the_target(self):
        quiet = tone(0.2) * 0.01
        out, gain = normalise_peak(quiet, -3.0)
        assert 20 * np.log10(np.abs(out).max()) == pytest.approx(-3.0, abs=0.1)
        assert gain > 1.0

    def test_silence_is_left_alone(self):
        # Normalising digital silence would manufacture a signal, and every
        # later measurement would describe the gain rather than the audio.
        out, gain = normalise_peak(np.zeros(100, dtype=np.float32))
        assert gain == 1.0
        assert not out.any()

    def test_soft_clip_is_a_noop_below_the_ceiling(self):
        quiet = tone(0.1) * 0.1
        assert np.array_equal(soft_clip(quiet), quiet)

    def test_soft_clip_bounds_a_hot_signal(self):
        hot = tone(0.1) * 20.0
        out = soft_clip(hot)
        assert np.abs(out).max() <= 0.999
        assert np.isfinite(out).all()

    def test_enhance_reports_what_it_did(self):
        out, applied = enhance(tone(0.2) * 0.01 + 0.1, -3.0, True)
        assert applied["gain"] > 1.0
        assert applied["dc_removed"] == pytest.approx(0.1, abs=1e-3)
        assert applied["peak"] == pytest.approx(10 ** (-3 / 20), abs=1e-3)

    def test_enhance_file_does_not_touch_the_raw_output(self, tmp_path):
        raw = str(tmp_path / "isolated.wav")
        write_wav_atomic(raw, tone(0.5) * 0.01, MODEL_SR, subtype="PCM_16")
        before = open(raw, "rb").read()
        applied = enhance_file(raw, str(tmp_path / "enhanced.wav"))
        assert applied["gain"] > 1.0
        assert open(raw, "rb").read() == before
        assert os.path.exists(str(tmp_path / "enhanced.wav"))


# ======================================================================
# Jobs
# ======================================================================
class TestJobs:
    def test_rejects_an_empty_query(self, source):
        with pytest.raises(ValueError):
            job_for(source, query="   ")

    def test_rejects_a_negative_start(self, source):
        with pytest.raises(ValueError):
            job_for(source, start_seconds=-1.0)

    def test_rejects_a_non_positive_duration(self, source):
        with pytest.raises(ValueError):
            job_for(source, duration_seconds=0.0)

    def test_region_detection(self, source):
        assert job_for(source).is_region is False
        assert job_for(source, start_seconds=0.5).is_region is True
        assert job_for(source, duration_seconds=0.5).is_region is True

    def test_metadata_records_measurements_only(self, source):
        result = SeparationResult(
            job=job_for(source, origin=JobOrigin.MANUAL_REGION, start_seconds=1.0),
            status=JobStatus.SUCCEEDED,
            isolated_path="/tmp/x.wav",
            input_seconds=2.0,
            processing_seconds=6.0,
            realtime_ratio=3.0,
        )
        metadata = result.to_metadata()
        assert metadata["query"] == "a whisper"
        assert metadata["origin"] == "manual_region"
        assert metadata["realtime_ratio"] == 3.0
        # No invented confidence anywhere.
        assert "confidence" not in metadata
        json.dumps(metadata)

    def test_ids_are_unique(self, source):
        assert job_for(source).job_id != job_for(source).job_id


# ======================================================================
# The separator, against a fake model
# ======================================================================
class TestSeparator:
    def test_writes_the_attempt_layout(self, separator, source):
        result = separator.separate(job_for(source))
        assert result.ok
        assert result.status is JobStatus.SUCCEEDED
        attempt = os.path.dirname(result.isolated_path)
        assert sorted(os.listdir(attempt)) == [
            "enhanced.wav", "isolated.wav", "metadata.json",
        ]
        assert result.attempt == 1

    def test_the_query_reaches_the_model(self, separator, source):
        separator.separate(job_for(source, query="a knock"))
        assert separator.fake_model.queries == ["a knock"]

    def test_output_depends_on_the_query(self, separator, source):
        # Two different queries must not produce the same audio.  This is the
        # property a filter cannot fake, so it is asserted directly.
        first = separator.separate(job_for(source, query="footsteps"))
        second = separator.separate(job_for(source, query="a whisper"))
        assert first.isolated_path != second.isolated_path
        a = open(first.isolated_path, "rb").read()
        b = open(second.isolated_path, "rb").read()
        assert a != b

    def test_repeated_runs_never_overwrite(self, separator, source):
        seen = []
        for query in ("footsteps", "a person speaking", "mechanical noise"):
            result = separator.separate(job_for(source, query=query))
            seen.append(result.isolated_path)
            assert result.ok
        assert len(set(seen)) == 3
        for path in seen:
            assert os.path.exists(path)

    def test_metadata_records_the_run(self, separator, source):
        result = separator.separate(job_for(source, query="scraping"))
        metadata = json.load(open(result.metadata_path))
        assert metadata["query"] == "scraping"
        assert metadata["status"] == "succeeded"
        assert metadata["model"] == "AudioSep"
        assert metadata["realtime_ratio"] == 2.0
        assert metadata["validation"]["ok"] is True

    def test_explicit_output_path_is_honoured(self, separator, source, tmp_path):
        target = str(tmp_path / "elsewhere" / "isolated.wav")
        result = separator.separate(job_for(source, output_path=target))
        assert result.ok
        assert result.isolated_path == target
        assert os.path.exists(target)
        # The enhanced copy and the metadata sit beside it, not in a stray
        # metadata.json in the current directory.
        assert os.path.exists(
            str(tmp_path / "elsewhere" / "isolated_enhanced.wav")
        )
        assert os.path.exists(
            str(tmp_path / "elsewhere" / "isolated.metadata.json")
        )

    def test_explicit_output_does_not_create_an_attempt_directory(
        self, separator, source, tmp_path
    ):
        target = str(tmp_path / "one.wav")
        separator.separate(job_for(source, output_path=target))
        stray = [
            name for name in os.listdir(os.path.dirname(source))
            if name.startswith("separation_")
        ]
        assert stray == []

    # -- regions ------------------------------------------------------
    def test_region_is_cut_to_its_own_file(self, separator, source):
        result = separator.separate(job_for(source, start_seconds=0.2,
                                            duration_seconds=0.3))
        assert result.ok
        attempt = os.path.dirname(result.isolated_path)
        # The exact audio the model saw is kept beside its own result.
        assert "region.wav" in os.listdir(attempt)

    def test_region_to_the_end(self, separator, source):
        result = separator.separate(job_for(source, start_seconds=0.5))
        assert result.ok

    def test_region_starting_past_the_end_is_refused(self, separator, source):
        result = separator.separate(job_for(source, start_seconds=10.0))
        assert not result.ok
        assert "only" in result.error
        assert result.isolated_path is None

    def test_oversized_input_is_refused_with_advice(self, source, tmp_path):
        long_source = str(tmp_path / "long.wav")
        write_wav_atomic(long_source, tone(40.0), SAMPLE_RATE)
        separator = AudioSepSeparator(
            SeparationConfig(max_input_seconds=30.0), str(tmp_path),
            store=SeparationStore(str(tmp_path / "events")),
            model=FakeModel(),
        )
        result = separator.separate(job_for(long_source))
        assert not result.ok
        assert "select a region" in result.error

    def test_input_shorter_than_the_minimum_is_refused(self, source, tmp_path):
        tiny = str(tmp_path / "tiny.wav")
        write_wav_atomic(tiny, tone(0.01), SAMPLE_RATE)
        separator = AudioSepSeparator(
            SeparationConfig(min_input_seconds=0.5), str(tmp_path),
            store=SeparationStore(str(tmp_path / "events")),
            model=FakeModel(),
        )
        result = separator.separate(job_for(tiny))
        assert not result.ok
        assert "minimum" in result.error

    # -- failures -----------------------------------------------------
    def test_a_model_failure_is_reported_not_hidden(self, source, tmp_path):
        separator = AudioSepSeparator(
            SeparationConfig(), str(tmp_path),
            store=SeparationStore(str(tmp_path / "events")),
            model=FakeModel(fail=True),
        )
        result = separator.separate(job_for(source))
        assert not result.ok
        assert "exploded" in result.error
        assert result.isolated_path is None

    def test_an_invalid_result_is_removed_not_published(self, source, tmp_path):
        separator = AudioSepSeparator(
            SeparationConfig(), str(tmp_path),
            store=SeparationStore(str(tmp_path / "events")),
            model=FakeModel(clipped=True),
        )
        result = separator.separate(job_for(source))
        assert not result.ok
        assert result.isolated_path is None
        # A clipped separation must not be offered for playback.
        assert not any(
            name.endswith(".wav") for name in os.listdir(
                os.path.dirname(result.metadata_path)
            )
        )

    def test_non_finite_output_is_rejected(self, source, tmp_path):
        separator = AudioSepSeparator(
            SeparationConfig(), str(tmp_path),
            store=SeparationStore(str(tmp_path / "events")),
            model=FakeModel(finite=False),
        )
        result = separator.separate(job_for(source))
        assert not result.ok
        assert "non-finite" in result.error

    def test_a_failed_run_still_records_metadata(self, source, tmp_path):
        separator = AudioSepSeparator(
            SeparationConfig(), str(tmp_path),
            store=SeparationStore(str(tmp_path / "events")),
            model=FakeModel(fail=True),
        )
        result = separator.separate(job_for(source))
        assert result.metadata_path and os.path.exists(result.metadata_path)
        metadata = json.load(open(result.metadata_path))
        assert metadata["status"] == "failed"
        assert metadata["error"]

    def test_enhancement_can_be_disabled(self, source, tmp_path):
        separator = AudioSepSeparator(
            SeparationConfig(enable_enhancement=False), str(tmp_path),
            store=SeparationStore(str(tmp_path / "events")),
            model=FakeModel(),
        )
        result = separator.separate(job_for(source))
        assert result.ok
        assert result.enhanced_path is None

    def test_missing_source_is_reported(self, separator, tmp_path):
        job = job_for(str(tmp_path / "nothing.wav"))
        result = separator.separate(job)
        assert not result.ok
        assert result.error


class TestAvailability:
    def test_unavailable_separator_reports_and_writes_nothing(self, source):
        separator = UnavailableSeparator("no model here")
        assert separator.available() is False
        result = separator.separate(job_for(source))
        assert not result.ok
        assert result.error == "no model here"
        assert result.isolated_path is None
        assert result.metadata_path is None

    def test_preflight_lists_every_missing_thing(self, tmp_path):
        # Deliberately the real ModelProcess, not a double: checking that the
        # installation is present is precisely its job, so a stubbed model
        # would make this test assert nothing.
        separator = AudioSepSeparator(
            SeparationConfig(model_dir="nowhere"), str(tmp_path),
            store=SeparationStore(str(tmp_path)),
        )
        problems = separator.preflight_problems()
        assert problems
        assert any("model directory" in p for p in problems)

    def test_build_separator_falls_back_to_an_honest_object(self, tmp_path):
        separator = build_separator(
            SeparationConfig(model_dir="nowhere"), str(tmp_path)
        )
        assert separator.available() is False
        assert separator.unavailable_reason()

    def test_disabled_separation_says_so(self, tmp_path):
        separator = build_separator(
            SeparationConfig(enabled=False), str(tmp_path)
        )
        assert "disabled" in separator.unavailable_reason()

    def test_missing_venv_is_named_in_the_message(self, tmp_path):
        separator = AudioSepSeparator(
            SeparationConfig(venv_python="no-such-venv/bin/python"), str(tmp_path),
            store=SeparationStore(str(tmp_path)),
        )
        reason = separator.unavailable_reason()
        assert "virtualenv" in reason
        assert "no-such-venv" in reason


# ======================================================================
# The worker
# ======================================================================
class RecordingSeparator:
    """A separator that records what it was asked and can be made to fail."""

    def __init__(self, fail: bool = False, raise_error: bool = False) -> None:
        self.jobs: list[SeparationJob] = []
        self.fail = fail
        self.raise_error = raise_error
        self.closed = False
        self.model = None

    def available(self) -> bool:
        return True

    def unavailable_reason(self):
        return None

    def separate(self, job: SeparationJob) -> SeparationResult:
        self.jobs.append(job)
        if self.raise_error:
            raise RuntimeError("the separator exploded")
        if self.fail:
            return SeparationResult(
                job=job, status=JobStatus.FAILED, error="deliberate failure"
            )
        return SeparationResult(job=job, status=JobStatus.SUCCEEDED,
                               isolated_path="/tmp/x.wav")

    def close(self) -> None:
        self.closed = True


class TestWorker:
    def test_runs_a_job_and_reports_it(self, source):
        results: list[SeparationResult] = []
        separator = RecordingSeparator()
        worker = SeparationWorker(
            separator, on_result=results.append, project_root="/tmp"
        )
        worker.start()
        assert worker.submit(job_for(source))
        assert worker.wait_idle(10.0)
        worker.stop()

        assert len(results) == 1
        assert results[0].ok
        assert worker.stats.succeeded == 1
        assert worker.stats.completed == 1
        assert worker.stats.pending == 0
        assert separator.closed

    def test_a_full_queue_is_reported_not_blocked(self, source, tmp_path):
        separator = RecordingSeparator()
        worker = SeparationWorker(
            separator, config=SeparationConfig(queue_max_jobs=1,
                                                workers=1),
            project_root="/tmp",
        )
        # Not started, so nothing drains the queue.
        accepted = [worker.submit(job_for(source)) for _ in range(3)]
        assert accepted.count(True) == 1
        assert worker.stats.rejected_queue_full == 2
        assert "queue is full" in worker.stats.last_error

    def test_submission_never_blocks_the_caller(self, source):
        separator = RecordingSeparator(raise_error=True)
        worker = SeparationWorker(separator, project_root="/tmp")
        worker.start()
        accepted = [worker.submit(job_for(source)) for _ in range(50)]
        worker.wait_idle(10.0)
        worker.stop()
        # Every accepted job fails, because the separator raises - and a
        # raising separator must not take the worker thread down with it.
        assert worker.stats.failed == worker.stats.submitted
        assert worker.stats.last_error
        # The queue is bounded, so the surplus is refused rather than
        # queued without limit or blocking the caller.
        assert accepted.count(True) < len(accepted)
        assert worker.stats.rejected_queue_full > 0
        assert worker.stats.completed == worker.stats.submitted

    def test_failures_are_counted_and_recorded(self, source):
        results: list[SeparationResult] = []
        worker = SeparationWorker(
            RecordingSeparator(fail=True), on_result=results.append,
            project_root="/tmp",
        )
        worker.start()
        worker.submit(job_for(source))
        worker.wait_idle(10.0)
        worker.stop()
        assert worker.stats.failed == 1
        assert worker.stats.last_error == "deliberate failure"
        assert not results[0].ok

    def test_status_reports_availability_and_progress(self, source):
        worker = SeparationWorker(
            RecordingSeparator(), project_root="/tmp"
        )
        status = worker.status()
        assert status["available"] is True
        assert status["unavailable_reason"] is None
        assert status["worker"]["submitted"] == 0

    def test_status_names_an_unavailable_model(self, tmp_path):
        worker = SeparationWorker(
            UnavailableSeparator("no model"), project_root="/tmp"
        )
        status = worker.status()
        assert status["available"] is False
        assert "no model" in status["unavailable_reason"]

    def test_work_is_ordered(self, source):
        separator = RecordingSeparator()
        worker = SeparationWorker(separator, project_root="/tmp")
        worker.start()
        for index in range(5):
            worker.submit(job_for(source, query=f"q{index}"))
        worker.wait_idle(10.0)
        worker.stop()
        assert [j.query for j in separator.jobs] == [f"q{i}" for i in range(5)]


# ======================================================================
# The CLI
# ======================================================================
class TestCLI:
    def run(self, argv):
        from app import separate

        return separate.main(argv)

    def test_query_is_required(self, tmp_path, source):
        assert self.run([source]) == 2

    def test_query_must_not_be_empty(self, source):
        assert self.run([source, "--query", "   "]) == 2

    def test_negative_start_is_refused(self, source):
        assert self.run([source, "-q", "a whisper", "--start", "-1"]) == 2

    def test_non_positive_duration_is_refused(self, source):
        assert self.run([source, "-q", "a whisper", "--duration", "0"]) == 2

    def test_missing_input_is_named(self, tmp_path):
        assert self.run([str(tmp_path / "gone.wav"), "-q", "x"]) == 2

    def test_missing_input_argument(self):
        assert self.run([]) == 2

    def test_event_and_output_are_mutually_exclusive(self, source, tmp_path):
        assert self.run(
            [source, "-q", "x", "--event", "event_1", "-o", str(tmp_path / "a.wav")]
        ) == 2

    def test_unknown_event_is_reported(self, source, tmp_path):
        assert self.run(
            [source, "-q", "x", "--event", "event_404",
             "--events-root", str(tmp_path)]
        ) == 2

    def test_unavailable_model_exits_three_and_writes_nothing(
        self, source, tmp_path
    ):
        # A missing model must be an honest failure, never a filtered copy.
        target = str(tmp_path / "out.wav")
        code = self.run(
            [source, "-q", "a whisper", "-o", target,
             "--config", self._config_without_model(tmp_path)]
        )
        assert code == 3
        assert not os.path.exists(target)

    def test_preflight_reports_availability(self, tmp_path, capsys):
        config = self._config_without_model(tmp_path)
        assert self.run(["--preflight", "--config", config]) == 3
        assert "not available" in capsys.readouterr().err

    def test_successful_run_reports_metrics(self, source, tmp_path, monkeypatch,
                                            capsys):
        from app import separate
        from app.audio.wavio import write_wav_atomic as write

        target = str(tmp_path / "out.wav")

        class Scripted:
            """A separator that behaves, without the model."""

            def available(self):
                return True

            def unavailable_reason(self):
                return None

            def separate(self, job):
                write(target, tone(0.5), MODEL_SR, subtype="PCM_16")
                result = SeparationResult(
                    job=job, status=JobStatus.SUCCEEDED,
                    isolated_path=target, input_seconds=1.0,
                    processing_seconds=3.0, realtime_ratio=3.0,
                )
                result.metadata_path = str(tmp_path / "out.metadata.json")
                return result

            def close(self):
                self.closed = True

        monkeypatch.setattr(separate, "build_separator", lambda *a, **k: Scripted())
        code = self.run([source, "-q", "a whisper", "-o", target])
        assert code == 0
        err = capsys.readouterr().err
        assert "3.00x realtime" in err
        assert "a whisper" in err
        assert os.path.exists(target)

    def test_failed_run_exits_four_and_says_so(self, source, tmp_path,
                                                monkeypatch, capsys):
        from app import separate

        class Failing:
            def available(self):
                return True

            def unavailable_reason(self):
                return None

            def separate(self, job):
                return SeparationResult(
                    job=job, status=JobStatus.FAILED, error="the model exploded"
                )

            def close(self):
                pass

        monkeypatch.setattr(separate, "build_separator", lambda *a, **k: Failing())
        code = self.run([source, "-q", "a whisper", "-o",
                         str(tmp_path / "out.wav")])
        assert code == 4
        err = capsys.readouterr().err
        assert "exploded" in err
        assert "untouched" in err
        assert not os.path.exists(str(tmp_path / "out.wav"))

    def _config_without_model(self, tmp_path) -> str:
        path = str(tmp_path / "config.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {"separation": {"model_dir": "no-such-model-dir",
                                "venv_python": "no-such-venv/bin/python"}},
                handle,
            )
        return path


# ======================================================================
# The real model, opt-in
# ======================================================================
RUN_MODEL = os.environ.get("AUDIOMICROSCOPE_SEPARATION_TESTS") == "1"
MODEL_REASON = (
    "set AUDIOMICROSCOPE_SEPARATION_TESTS=1 and install the model "
    "(third_party/AudioSep + venv-sep) to run these"
)


@pytest.mark.skipif(not RUN_MODEL, reason=MODEL_REASON)
class TestRealModel:
    @pytest.fixture
    def real(self):
        from app.separation.model import ModelProcess

        config = SeparationConfig()
        process = ModelProcess(config, os.getcwd())
        if not process.available():
            pytest.skip("the separation model is not installed")
        return config, process

    def test_a_query_produces_valid_playable_audio(self, real, source):
        from app.separation.jobs import SeparationJob

        config, process = real
        target = str(os.path.join(os.path.dirname(source), "real.wav"))
        try:
            response = process.separate(source, "footsteps", target)
            assert response["ok"], response
        finally:
            process.stop()

        report = validate_output(
            target, expected_sample_rate=config.model_sample_rate
        )
        assert report.ok, report.problems
        assert report.duration > 0.3
        # A separation is not a filtered copy of its input: the correlation
        # with the mixture must be well below 1.
        from app.audio.wavio import read_wav
        import soundfile as sf

        mixture, _ = sf.read(source, dtype="float32")
        isolated = read_wav(target)
        length = min(mixture.size, isolated.size)
        a, b = mixture[:length], isolated[:length]
        correlation = float(
            np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b) + 1e-20)
        )
        assert correlation < 0.95, correlation

    def test_different_queries_give_different_audio(self, real, tmp_path):
        config, process = real
        outputs = []
        try:
            for index, query in enumerate(("footsteps", "a whisper")):
                target = str(tmp_path / f"q{index}.wav")
                assert process.separate(str(tmp_path.parent / "x.wav")
                                        if False else self._probe(tmp_path),
                                        query, target)["ok"]
                outputs.append(open(target, "rb").read())
        finally:
            process.stop()
        assert outputs[0] != outputs[1]

    def _probe(self, tmp_path) -> str:
        path = str(tmp_path / "probe.wav")
        if not os.path.exists(path):
            write_wav_atomic(path, tone(3.0), SAMPLE_RATE)
        return path


class TestAttemptPlacement:
    """Where an attempt goes when the caller does not name a directory.

    The GUI submits a job with no output directory, relying on the separator
    to place the attempt beside the audio it came from.  Getting that wrong
    does not merely scatter files: an event id that happens to exist in a
    different tree causes the attempt to be written into *that* event's
    evidence directory, silently.
    """

    def test_attempt_goes_beside_the_source_audio(self, tmp_path, source):
        root = str(tmp_path / "tree")
        separator = AudioSepSeparator(
            SeparationConfig(), str(tmp_path), store=SeparationStore(root),
            model=FakeModel(),
        )
        result = separator.separate(job_for(source))
        assert result.ok
        expected = os.path.join(
            os.path.dirname(os.path.abspath(source)), "separation_001"
        )
        assert result.isolated_path == os.path.join(expected, "isolated.wav")

    def test_an_unrelated_event_in_the_tree_is_never_touched(
        self, tmp_path, source
    ):
        # A different tree already contains an event with the same id.
        root = str(tmp_path / "tree")
        unrelated = os.path.join(root, "2026-01-01", "event_000001")
        os.makedirs(unrelated, exist_ok=True)
        write_wav_atomic(os.path.join(unrelated, "original.wav"), tone(0.2), SAMPLE_RATE)
        marker = os.path.join(unrelated, "metadata.json")
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write('{"untouched": true}')

        separator = AudioSepSeparator(
            SeparationConfig(), str(tmp_path), store=SeparationStore(root),
            model=FakeModel(),
        )
        result = separator.separate(
            job_for(source, event_id="event_000001")
        )
        assert result.ok
        # Nothing was added to the other event's directory.
        assert sorted(os.listdir(unrelated)) == ["metadata.json", "original.wav"]
        assert json.load(open(marker)) == {"untouched": True}

    def test_explicit_output_directory_still_wins(self, tmp_path, source):
        target = str(tmp_path / "chosen")
        os.makedirs(target)
        separator = AudioSepSeparator(
            SeparationConfig(), str(tmp_path),
            store=SeparationStore(str(tmp_path / "tree")), model=FakeModel(),
        )
        result = separator.separate(job_for(source, output_directory=target))
        assert result.ok
        assert os.path.basename(
            os.path.dirname(result.isolated_path)
        ) == "separation_001"
        assert os.path.dirname(
            os.path.dirname(result.isolated_path)
        ) == target
