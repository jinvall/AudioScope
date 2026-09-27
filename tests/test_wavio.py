"""WAV I/O: losslessness and atomic writes (docs/DATA_AND_STORAGE.md 3, 9)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from app.audio.wavio import (
    WavWriter,
    probe_wav,
    read_wav,
    read_wav_mono,
    write_wav_atomic,
)


def test_round_trip_is_bit_exact(tmp_path):
    """Evidence must not be altered: float32 WAV stores samples exactly."""
    data = (np.random.default_rng(0).standard_normal(48000) * 0.3).astype(
        np.float32
    )
    path = str(tmp_path / "e.wav")
    write_wav_atomic(path, data, 48000)
    back = read_wav_mono(path)
    assert np.array_equal(data, back)


def test_probe_reports_real_format(tmp_path):
    path = str(tmp_path / "e.wav")
    data = np.zeros(48000, dtype=np.float32)
    write_wav_atomic(path, data, 48000)
    info = probe_wav(path)
    assert info.sample_rate == 48000
    assert info.channels == 1
    assert info.frames == 48000
    assert info.duration == pytest.approx(1.0)
    assert info.subtype == "FLOAT"
    assert info.is_mono


def test_extreme_values_are_not_clipped(tmp_path):
    """Values beyond +/-1.0 must survive; clipping would destroy evidence."""
    data = np.array([2.5, -3.0, 0.0, 1.0, -1.0], dtype=np.float32)
    path = str(tmp_path / "loud.wav")
    write_wav_atomic(path, data, 48000)
    back = read_wav_mono(path)
    assert np.array_equal(data, back)


def test_partial_read(tmp_path):
    data = np.arange(10000, dtype=np.float32)
    path = str(tmp_path / "e.wav")
    write_wav_atomic(path, data, 48000)
    segment = read_wav(path, start=1000, frames=500)
    assert segment.size == 500
    assert segment[0] == 1000.0
    assert segment[-1] == 1499.0


def test_read_past_end_raises(tmp_path):
    path = str(tmp_path / "e.wav")
    write_wav_atomic(path, np.zeros(100, dtype=np.float32), 48000)
    with pytest.raises(ValueError):
        read_wav(path, start=500, frames=10)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_wav_mono(str(tmp_path / "nope.wav"))
    with pytest.raises(FileNotFoundError):
        probe_wav(str(tmp_path / "nope.wav"))


def test_no_partial_file_left_on_success(tmp_path):
    path = str(tmp_path / "e.wav")
    write_wav_atomic(path, np.zeros(1000, dtype=np.float32), 48000)
    leftovers = [f for f in os.listdir(tmp_path) if f.endswith(".part")]
    assert leftovers == []


def test_failed_write_leaves_no_partial_final_file(tmp_path, monkeypatch):
    """An interrupted write must not leave a truncated file that looks valid."""
    import app.audio.wavio as wavio

    path = str(tmp_path / "e.wav")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(wavio.sf, "SoundFile", boom)
    with pytest.raises(OSError):
        write_wav_atomic(path, np.zeros(1000, dtype=np.float32), 48000)

    assert not os.path.exists(path), "a corrupt final file was left behind"
    leftovers = os.listdir(tmp_path)
    assert leftovers == [], f"temporary files not cleaned up: {leftovers}"


def test_creates_missing_directories(tmp_path):
    path = str(tmp_path / "a" / "b" / "c" / "e.wav")
    write_wav_atomic(path, np.zeros(100, dtype=np.float32), 48000)
    assert os.path.exists(path)


# ----------------------------------------------------------------------
# WavWriter chunking
# ----------------------------------------------------------------------
def test_writer_rolls_over_at_chunk_length(tmp_path):
    writer = WavWriter(str(tmp_path), 48000, chunk_frames=4800)
    block = np.zeros(1024, dtype=np.float32)
    completed = []
    for _ in range(10):
        done = writer.append(block)
        if done:
            completed.append(done)
    writer.close()

    assert len(completed) == 2
    for path in completed:
        assert probe_wav(path).frames == 4800


def test_writer_does_not_drop_the_overshoot(tmp_path):
    """Samples past a chunk boundary must carry into the next chunk."""
    rate, chunk = 48000, 4800
    data = np.arange(10000, dtype=np.float32)
    writer = WavWriter(str(tmp_path), rate, chunk_frames=chunk)
    for i in range(0, data.size, 1000):
        writer.append(data[i:i + 1000])
    writer.close()

    paths = sorted(tmp_path.glob("*.wav"))
    recovered = np.concatenate([read_wav_mono(str(p)) for p in paths])
    assert recovered.size == data.size
    assert np.array_equal(recovered, data), "audio was lost at the chunk seam"


def test_writer_round_trip_is_exact(tmp_path):
    data = (np.random.default_rng(3).standard_normal(20000) * 0.2).astype(
        np.float32
    )
    writer = WavWriter(str(tmp_path), 48000, chunk_frames=7000)
    for i in range(0, data.size, 999):
        writer.append(data[i:i + 999])
    writer.close()
    paths = sorted(tmp_path.glob("*.wav"))
    recovered = np.concatenate([read_wav_mono(str(p)) for p in paths])
    assert np.array_equal(recovered, data)


def test_writer_close_with_nothing_buffered(tmp_path):
    writer = WavWriter(str(tmp_path), 48000, chunk_frames=4800)
    assert writer.close() is None
    assert writer.append(np.zeros(0, np.float32)) is None
    assert list(tmp_path.glob("*.wav")) == []


def test_writer_rejects_bad_chunk_length(tmp_path):
    with pytest.raises(ValueError):
        WavWriter(str(tmp_path), 48000, chunk_frames=0)
