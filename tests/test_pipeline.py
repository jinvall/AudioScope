"""Pipeline, file source, and recorder behaviour."""

from __future__ import annotations

import os

import numpy as np
import pytest

from app.audio.recorder import FileSource, Recorder
from app.audio.wavio import probe_wav, read_wav_mono, write_wav_atomic
from app.config import AppConfig
from app.pipeline import AudioPipeline


# ----------------------------------------------------------------------
# FileSource
# ----------------------------------------------------------------------
def test_file_source_reads_the_whole_file(config, tone, tmp_path):
    path = str(tmp_path / "a.wav")
    data = tone(440.0, 0.5)
    write_wav_atomic(path, data, 48000)

    source = FileSource(path, config)
    total = 0
    while True:
        block = source.read()
        if block is None:
            break
        total += block.size
    assert total == data.size


def test_file_source_streams_identical_samples(config, tone, tmp_path):
    """Blocks must reassemble into the original file, bit for bit."""
    path = str(tmp_path / "a.wav")
    data = tone(1000.0, 0.25)
    write_wav_atomic(path, data, 48000)

    source = FileSource(path, config)
    blocks = []
    while True:
        block = source.read()
        if block is None:
            break
        blocks.append(block)
    assert np.array_equal(np.concatenate(blocks), data)


def test_file_source_downmixes_stereo(config, tone, tmp_path):
    path = str(tmp_path / "stereo.wav")
    left = tone(440.0, 0.1)
    write_wav_atomic(path, np.stack([left, left], axis=1), 48000)
    source = FileSource(path, config)
    assert source.native_sample_rate == 48000
    assert source.read() is not None


def test_file_source_honours_block_size(tmp_path, tone):
    config = AppConfig(audio={"block_size": 256}).validate()
    path = str(tmp_path / "a.wav")
    write_wav_atomic(path, tone(440.0, 0.2), 48000)
    source = FileSource(path, config)
    assert source.read().size == 256


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
def test_pipeline_feeds_ring_and_records(config, tone, tmp_path):
    path = str(tmp_path / "a.wav")
    data = tone(1000.0, 1.0)
    write_wav_atomic(path, data, 48000)

    out_dir = str(tmp_path / "rec")
    source = FileSource(path, config)
    pipeline = AudioPipeline(config, source, record=True, record_directory=out_dir)
    pipeline.start()
    pipeline.wait()
    stats = pipeline.stop()

    assert stats.frames == data.size
    assert pipeline.ring.frames_available == data.size
    assert pipeline.ring.read(0, data.size)[100] == pytest.approx(
        data[100], rel=1e-6
    )

    rec = stats and pipeline.recorder.stats
    assert rec.frames_written == data.size
    assert rec.is_contiguous


def test_pipeline_recorded_file_matches_input(config, tone, tmp_path):
    path = str(tmp_path / "a.wav")
    data = tone(440.0, 0.5)
    write_wav_atomic(path, data, 48000)

    out_dir = str(tmp_path / "rec")
    pipeline = AudioPipeline(
        config, FileSource(path, config), record=True, record_directory=out_dir
    )
    pipeline.start()
    pipeline.wait()
    pipeline.stop()

    chunks = sorted(
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".wav")
    )
    assert chunks
    recovered = np.concatenate([read_wav_mono(p) for p in chunks])
    assert np.array_equal(recovered, data), "recording is not bit-exact"


def test_pipeline_without_recording_writes_nothing(config, tone, tmp_path):
    path = str(tmp_path / "a.wav")
    write_wav_atomic(path, tone(440.0, 0.2), 48000)
    out_dir = str(tmp_path / "rec")
    pipeline = AudioPipeline(
        config, FileSource(path, config), record=False, record_directory=out_dir
    )
    pipeline.start()
    pipeline.wait()
    pipeline.stop()
    assert not os.path.exists(out_dir) or os.listdir(out_dir) == []


def test_on_block_hook_receives_every_block(config, tone, tmp_path):
    """The hook Phase 2 attaches must actually see the audio."""
    path = str(tmp_path / "a.wav")
    data = tone(1000.0, 0.1)
    write_wav_atomic(path, data, 48000)

    seen: list[np.ndarray] = []
    pipeline = AudioPipeline(
        config, FileSource(path, config), record=False
    )
    pipeline.on_block = lambda pos, block: seen.append(block.copy())
    pipeline.start()
    pipeline.wait()
    pipeline.stop()

    assert seen
    assert sum(b.size for b in seen) == data.size
    assert np.array_equal(np.concatenate(seen), data)


def test_failing_hook_does_not_stop_capture(config, tone, tmp_path):
    """A broken consumer must not take the capture path down (AGENTS.md 9)."""
    path = str(tmp_path / "a.wav")
    data = tone(1000.0, 0.1)
    write_wav_atomic(path, data, 48000)

    def explode(pos, block):
        raise RuntimeError("consumer bug")

    pipeline = AudioPipeline(config, FileSource(path, config), record=False)
    pipeline.on_block = explode
    pipeline.start()
    pipeline.wait()
    stats = pipeline.stop()

    assert stats.frames == data.size
    assert any("consumer bug" in e for e in stats.errors)


def test_realtime_ratio_is_reported(config, tone, tmp_path):
    path = str(tmp_path / "a.wav")
    write_wav_atomic(path, tone(440.0, 2.0), 48000)
    pipeline = AudioPipeline(config, FileSource(path, config), record=False)
    pipeline.start()
    pipeline.wait()
    stats = pipeline.stop()
    ratio = stats.realtime_ratio
    assert ratio > 0
    # File replay should be much faster than real time.
    assert ratio < 1.0, f"expected faster than realtime, got {ratio}"


def test_summary_is_serialisable(config, tone, tmp_path):
    import json

    path = str(tmp_path / "a.wav")
    write_wav_atomic(path, tone(440.0, 0.2), 48000)
    pipeline = AudioPipeline(
        config, FileSource(path, config), record=True,
        record_directory=str(tmp_path / "rec"),
    )
    pipeline.start()
    pipeline.wait()
    pipeline.stop()
    text = json.dumps(pipeline.summary())
    assert "sample_rate" in text
    assert "recorder" in text


# ----------------------------------------------------------------------
# Recorder
# ----------------------------------------------------------------------
def test_recorder_rolls_chunks(config):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        # 1 s chunks: the smallest chunk length the configuration allows.
        small = AppConfig(record={"chunk_seconds": 1.0}).validate()
        rec = Recorder(small, directory=tmp)
        rec.start()
        block = np.zeros(4800, dtype=np.float32)
        for _ in range(60):  # 288 000 frames = 6 chunks at 48 000 frames each
            assert rec.write(block)
        stats = rec.stop()
        assert len(stats.chunks_written) >= 6
        assert stats.frames_written == 60 * 4800
        assert stats.is_contiguous


def test_recorder_reports_drops_rather_than_hiding_them(config):
    """A full queue must be counted and reported, never silently swallowed."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cfg = AppConfig(
            audio={"queue_max_blocks": 2},
            record={"chunk_seconds": 1.0},
        ).validate()
        rec = Recorder(cfg, directory=tmp)
        rec.start()
        block = np.zeros(4800, dtype=np.float32)
        accepted = sum(1 for _ in range(50) if rec.write(block))
        stats = rec.stop()
        assert stats.blocks_dropped > 0
        assert not stats.is_contiguous
        assert accepted < 50
        # Contiguity is reported honestly in the serialised form too.
        assert stats.to_dict()["is_contiguous"] is False
