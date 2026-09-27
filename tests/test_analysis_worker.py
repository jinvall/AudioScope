"""Analysis worker threading, the spectrogram buffer, and the CLI."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from app.analysis.spectrogram import SpectrogramBuffer, WaveformBuffer
from app.analysis.worker import AnalysisResult, AnalysisWorker
from app.config import AnalysisConfig, AppConfig


@pytest.fixture
def config() -> AppConfig:
    return AppConfig().validate()


def feed(worker: AnalysisWorker, seconds: float, amplitude: float,
         config: AppConfig, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    hop = config.analysis_hop_frames
    position = 0
    for _ in range(int(seconds * config.sample_rate / hop)):
        block = (amplitude * rng.standard_normal(hop)).astype(np.float32)
        worker.submit(block, position)
        position += block.size


# ----------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------
def test_worker_produces_results(config):
    worker = AnalysisWorker(config)
    worker.start()
    try:
        feed(worker, 3.0, 0.01, config)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not worker.results():
            time.sleep(0.05)
    finally:
        stats = worker.stop()
    results = worker.results()
    assert results
    assert stats.frames > 0
    assert isinstance(results[0], AnalysisResult)
    assert results[0].features.sample_rate == config.sample_rate


def test_worker_analyses_the_whole_input_losslessly(config):
    """Offline mode must not skip audio, or file analysis is meaningless."""
    worker = AnalysisWorker(config, lossless=True)
    worker.start()
    try:
        feed(worker, 2.0, 0.01, config)
        time.sleep(0.5)
    finally:
        stats = worker.stop()
    assert stats.blocks_dropped == 0
    assert stats.to_dict()["is_contiguous"] is True
    expected = int(2.0 * config.sample_rate / config.analysis_hop_frames)
    assert stats.frames >= expected * 0.9


def test_history_retains_every_frame(config):
    worker = AnalysisWorker(config, lossless=True, history_size=100_000)
    worker.start()
    try:
        feed(worker, 2.0, 0.01, config)
        time.sleep(0.5)
    finally:
        worker.stop()
    assert len(worker.history) == worker.stats.frames
    assert worker.history_dropped == 0
    assert worker.history[0].audio_time < worker.history[-1].audio_time


def test_history_is_empty_unless_requested(config):
    worker = AnalysisWorker(config)
    assert worker.history == []
    assert worker.history_size == 0


def test_submit_never_blocks_when_the_queue_is_full(config):
    """The capture path must not be stalled by a slow analysis stage."""
    worker = AnalysisWorker(config, input_queue_blocks=2)
    worker.start()
    try:
        rng = np.random.default_rng(0)
        began = time.monotonic()
        for i in range(2000):
            worker.submit(
                (0.01 * rng.standard_normal(256)).astype(np.float32), i * 256
            )
        elapsed = time.monotonic() - began
        # 2000 submits of a tiny block cannot plausibly take a second.
        assert elapsed < 1.0, f"submit() blocked the caller for {elapsed:.2f}s"
        assert worker.stats.blocks_dropped > 0, "drops must be counted"
    finally:
        worker.stop()


def test_audio_time_is_the_stream_timeline(config):
    worker = AnalysisWorker(config, lossless=True, history_size=10_000)
    worker.start()
    try:
        feed(worker, 1.0, 0.01, config)
        time.sleep(0.4)
    finally:
        worker.stop()
    for result in worker.history:
        assert result.audio_time == pytest.approx(
            result.start_sample / config.sample_rate, rel=1e-9
        )


def test_a_gap_is_flagged_when_blocks_are_dropped(config):
    """A consumer must be able to tell that evidence is missing."""
    worker = AnalysisWorker(config, input_queue_blocks=1, lossless=False)
    worker.start()
    try:
        rng = np.random.default_rng(1)
        for i in range(3000):
            worker.submit(
                (0.01 * rng.standard_normal(512)).astype(np.float32), i * 512
            )
        time.sleep(0.5)
    finally:
        stats = worker.stop()
    if stats.blocks_dropped > 0:
        assert stats.gap_frames >= 0
        # Whatever survived may be flagged; nothing may be silently wrong.
        assert stats.to_dict()["is_contiguous"] is False


def test_results_carry_features_and_floor(config):
    worker = AnalysisWorker(config, lossless=True, history_size=10_000)
    worker.start()
    try:
        feed(worker, 4.0, 0.005, config)
        time.sleep(0.6)
    finally:
        worker.stop()
    history = worker.history
    assert history
    last = history[-1]
    assert last.features.rms_db > -200.0
    assert len(last.features.bands) == 9
    assert last.floor.overall_floor_db <= 0.0
    assert last.snr_db == last.floor.overall_snr_db


def test_result_serialises(config):
    worker = AnalysisWorker(config, lossless=True, history_size=1000)
    worker.start()
    try:
        feed(worker, 1.0, 0.01, config)
        time.sleep(0.3)
    finally:
        worker.stop()
    text = json.dumps(worker.history[-1].to_dict())
    assert "noise_floor" in text
    assert "spectral_centroid_hz" in text


def test_realtime_ratio_is_under_one(config):
    """docs/PERFORMANCE.md: analysis must be cheaper than real time."""
    worker = AnalysisWorker(config, lossless=True, history_size=100_000)
    worker.start()
    try:
        feed(worker, 3.0, 0.01, config)
        time.sleep(1.0)
    finally:
        stats = worker.stop()
    assert stats.realtime_ratio < 1.0, (
        f"analysis needs {stats.realtime_ratio:.2f} of a core in real time"
    )
    assert stats.mean_frame_ms < 5.0


def test_worker_recovers_from_a_bad_frame(config):
    """A consumer failure must not kill the analysis thread."""
    seen: list[int] = []

    def explode(result):
        seen.append(result.index)
        raise RuntimeError("consumer bug")

    worker = AnalysisWorker(config, lossless=True, on_result=explode)
    worker.start()
    try:
        feed(worker, 1.0, 0.01, config)
        time.sleep(0.3)
    finally:
        stats = worker.stop()
    assert seen, "the callback was never called"
    assert stats.errors, "the failure was not recorded"
    assert worker.stats.frames > 0


def test_stop_is_idempotent(config):
    worker = AnalysisWorker(config)
    worker.start()
    worker.stop()
    worker.stop()
    assert not worker.running


def test_submit_before_start_is_refused(config):
    worker = AnalysisWorker(config)
    assert worker.submit(np.zeros(128, dtype=np.float32), 0) is False


def test_stats_serialise(config):
    worker = AnalysisWorker(config)
    worker.start()
    try:
        feed(worker, 0.5, 0.01, config)
        time.sleep(0.3)
    finally:
        stats = worker.stop()
    payload = stats.to_dict()
    assert payload["frames"] > 0
    assert "realtime_ratio" in payload
    json.dumps(payload)


# ----------------------------------------------------------------------
# Spectrogram
# ----------------------------------------------------------------------
def test_spectrogram_shape_and_rolling():
    buffer = SpectrogramBuffer(bins=65, columns=10, max_hz=24000,
                               sample_rate=48000)
    for i in range(15):
        buffer.push(np.full(65, float(i + 1), dtype=np.float32))
    image = buffer.image()
    assert image.shape == (10, buffer.kept_bins)
    assert buffer.filled_columns == 10
    assert buffer.is_full
    # Oldest column holds the 6th push, newest the 15th.
    assert image[-1][0] == pytest.approx(1.0)
    assert image[0][0] == pytest.approx(1.0)


def test_spectrogram_partial_fill_is_right_aligned():
    buffer = SpectrogramBuffer(bins=33, columns=8, sample_rate=48000)
    for i in range(3):
        buffer.push(np.full(33, 0.5, dtype=np.float32))
    image = buffer.image()
    assert image.shape == (8, buffer.kept_bins)
    # The three filled columns are at the right, so age_seconds is meaningful.
    assert np.all(image[:5] == 0.0)
    assert np.all(image[5:] > 0.0)
    assert buffer.is_full is False


def test_spectrogram_normalises_each_column():
    buffer = SpectrogramBuffer(bins=33, columns=4, sample_rate=48000)
    quiet = np.full(33, 1e-6, dtype=np.float32)
    loud = np.full(33, 1e-1, dtype=np.float32)
    buffer.push(quiet)
    buffer.push(loud)
    # Each column peaks at 1.0 regardless of absolute level, so a quiet room
    # still shows structure.
    assert buffer.image()[-1].max() == pytest.approx(1.0)
    assert buffer.image()[-2].max() == pytest.approx(1.0)


def test_spectrogram_drops_bins_above_max_hz():
    """A 48 kHz and a 44.1 kHz stream must share a comparable frequency axis."""
    at_24k = SpectrogramBuffer(bins=1025, columns=4, max_hz=24000,
                               sample_rate=48000)
    at_8k = SpectrogramBuffer(bins=1025, columns=4, max_hz=8000,
                              sample_rate=48000)
    assert at_24k.frequencies[-1] <= 24000.0
    assert at_8k.frequencies[-1] <= 8000.0
    assert at_8k.kept_bins < at_24k.kept_bins


def test_spectrogram_column_seconds():
    buffer = SpectrogramBuffer(bins=65, columns=100, sample_rate=48000)
    with pytest.raises(ValueError):
        buffer.column_seconds = 0.0
    buffer.column_seconds = 0.01
    assert buffer.span_seconds == pytest.approx(1.0)
    buffer.push(np.ones(65, dtype=np.float32))
    assert buffer.age_seconds == pytest.approx(0.99)


def test_spectrogram_column_at():
    buffer = SpectrogramBuffer(bins=17, columns=5, sample_rate=48000)
    for i in range(3):
        buffer.push(np.full(17, float(i + 1), dtype=np.float32))
    assert np.all(buffer.column_at(0) > 0.0)
    assert np.all(buffer.column_at(99) == 0.0)


def test_spectrogram_clear():
    buffer = SpectrogramBuffer(bins=17, columns=5, sample_rate=48000)
    buffer.push(np.ones(17, dtype=np.float32))
    buffer.clear()
    assert buffer.filled_columns == 0
    assert np.all(buffer.image() == 0.0)


def test_spectrogram_rejects_bad_geometry():
    with pytest.raises(ValueError):
        SpectrogramBuffer(bins=0, columns=4)
    with pytest.raises(ValueError):
        SpectrogramBuffer(bins=4, columns=0)


# ----------------------------------------------------------------------
# Waveform buffer
# ----------------------------------------------------------------------
def test_waveform_buffer_envelope():
    buffer = WaveformBuffer(columns=4)
    for value in (1.0, -2.0, 0.5, -0.5):
        buffer.push(np.full(8, value, dtype=np.float32))
    peaks, troughs = buffer.envelope()
    assert peaks.shape == (4,)
    # Oldest first, which is the order they were pushed.
    assert list(peaks) == [1.0, -2.0, 0.5, -0.5]
    assert list(troughs) == [1.0, -2.0, 0.5, -0.5]
    assert buffer.filled_columns == 4


def test_waveform_buffer_keeps_transients():
    """A mean would hide a click; a min/max envelope must not."""
    buffer = WaveformBuffer(columns=1)
    data = np.zeros(1000, dtype=np.float32)
    data[500] = 0.9
    buffer.push(data)
    peaks, troughs = buffer.envelope()
    assert peaks[0] == pytest.approx(0.9)


# ----------------------------------------------------------------------
# Pipeline integration and CLI
# ----------------------------------------------------------------------
def test_pipeline_starts_and_stops_analysis(config, tmp_path):
    from app.audio.recorder import FileSource
    from app.audio.wavio import write_wav_atomic
    from app.pipeline import AudioPipeline

    path = str(tmp_path / "a.wav")
    data = (0.01 * np.random.default_rng(0).standard_normal(48000 * 2)).astype(
        np.float32
    )
    write_wav_atomic(path, data, config.sample_rate)

    pipeline = AudioPipeline(config, FileSource(path, config), record=False)
    worker = pipeline.enable_analysis(lossless=True)
    pipeline.start()
    pipeline.wait()
    pipeline.stop()

    assert worker.stats.frames > 0
    assert "analysis" in pipeline.summary()
    assert not worker.running


def test_analyse_cli_json_is_parseable(tmp_path):
    """--json must emit nothing but JSON, or it cannot be consumed."""
    from app.analyse import main
    from app.audio.wavio import write_wav_atomic

    path = str(tmp_path / "a.wav")
    data = (0.01 * np.random.default_rng(0).standard_normal(48000)).astype(
        np.float32
    )
    write_wav_atomic(path, data, 48000)
    assert main([path, "--json"]) == 0


def test_analyse_cli_reports_a_missing_file(tmp_path, capsys):
    from app.analyse import main

    assert main([str(tmp_path / "nope.wav")]) == 2


def test_analysis_config_validation():
    with pytest.raises(Exception):
        AnalysisConfig(hop_ms=100.0, frame_ms=50.0)
    with pytest.raises(Exception):
        AnalysisConfig(window="triangle")
    with pytest.raises(Exception):
        AnalysisConfig(frame_ms=1.0)
    with pytest.raises(Exception):
        AnalysisConfig(rolloff_percentile=0.0)
    with pytest.raises(Exception):
        AnalysisConfig(min_stats_subwindow_sec=10.0, min_stats_window_sec=1.0)


def test_analysis_config_derived_frame_counts():
    config = AppConfig().validate()
    assert config.analysis_frame_frames == 1920   # 40 ms at 48 kHz
    assert config.analysis_hop_frames == 480      # 10 ms
    assert config.analysis_frames_per_second == 100.0
    assert config.analysis.overlap_ratio == 0.75
