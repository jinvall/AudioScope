"""Ring buffer: the pre-roll guarantee (docs/AUDIO_PIPELINE.md section 9)."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from app.audio.ringbuffer import RingBuffer


def test_capacity_matches_duration():
    rb = RingBuffer(48000, 30.0)
    assert rb.capacity == 48000 * 30
    assert len(rb) == rb.capacity


def test_write_then_read_preserves_samples_exactly():
    rb = RingBuffer(48000, 1.0)
    data = np.arange(1000, dtype=np.float32) / 1000.0
    rb.write(data)
    out = rb.read(0, 1000)
    assert np.array_equal(out, data)


def test_absolute_timeline_survives_wrap():
    """Reads are indexed by absolute frame, not by buffer position."""
    rb = RingBuffer(48000, 1.0)  # 48000 frames
    total = 0
    for i in range(10):
        block = np.full(10000, float(i), dtype=np.float32)
        rb.write(block)
        total += block.size
    assert total == 100_000
    # Newest 1000 frames are the last block written (i == 9).
    out = rb.read(total - 1000, total)
    assert np.all(out == 9.0)
    assert rb.total_written == 100_000


def test_oldest_data_is_discarded_past_capacity():
    rb = RingBuffer(48000, 1.0)
    rb.write(np.full(60_000, 7.0, dtype=np.float32))
    assert rb.frames_available == 48_000
    assert rb.oldest_available == 12_000
    # Requesting a range entirely in the past yields nothing, not stale data.
    assert rb.read(0, 10_000).size == 0
    assert rb.read(12_000, 12_010).size == 10


def test_pre_roll_returns_history_before_a_frame():
    """The whole point of the buffer: audio from before the event exists."""
    rb = RingBuffer(48000, 1.0)
    # 40 000 frames of distinguishable data, then the "event" frame at 40 000.
    ramp = np.arange(40_000, dtype=np.float32)
    rb.write(ramp)
    at_frame = 40_000
    pre, missing = rb.pre_roll(at_frame, 4_800)  # 100 ms of history
    assert missing == 0
    assert pre.size == 4_800
    assert pre[0] == 35_200.0
    assert pre[-1] == 39_999.0


def test_pre_roll_reports_missing_history():
    """A pre-roll longer than the retained window must be reported, not faked."""
    rb = RingBuffer(48000, 0.1)  # 4 800 frames
    rb.write(np.zeros(50_000, dtype=np.float32))
    # Ask for 8 000 frames of history; only 4 800 can still exist.
    pre, missing = rb.pre_roll(50_000, 8_000)
    assert missing == 8_000 - 4_800
    assert pre.size == 4_800


def test_pre_roll_fully_available_is_reported_as_such():
    """Requesting exactly what is retained must report zero missing."""
    rb = RingBuffer(48000, 1.0)  # 48 000 frames
    rb.write(np.zeros(50_000, dtype=np.float32))
    pre, missing = rb.pre_roll(50_000, 48_000)
    assert missing == 0
    assert pre.size == 48_000


def test_write_larger_than_capacity_keeps_the_tail():
    rb = RingBuffer(48000, 1.0)
    data = np.arange(60_000, dtype=np.float32)
    rb.write(data)
    out = rb.read(rb.oldest_available, rb.total_written)
    assert out.size == 48_000
    assert out[0] == 12_000.0
    assert out[-1] == 59_999.0
    assert rb.overruns > 0


def test_read_is_clamped_to_retained_window():
    rb = RingBuffer(48000, 1.0)
    rb.write(np.arange(10_000, dtype=np.float32))
    out = rb.read(5_000, 50_000)  # asks for far more than exists
    assert out.size == 5_000
    assert out[-1] == 9_999.0


def test_read_latest_and_snapshot():
    rb = RingBuffer(48000, 1.0)
    rb.write(np.arange(10_000, dtype=np.float32))
    latest = rb.read_latest(100)
    assert latest[0] == 9_900.0
    snap = rb.snapshot()
    assert snap.size == 10_000
    assert snap[0] == 0.0


def test_read_empty_range_returns_empty():
    rb = RingBuffer(48000, 1.0)
    rb.write(np.ones(100, dtype=np.float32))
    assert rb.read(50, 50).size == 0
    assert rb.read(80, 20).size == 0


def test_clear_resets_timeline():
    rb = RingBuffer(48000, 1.0)
    rb.write(np.ones(1000, dtype=np.float32))
    rb.clear()
    assert rb.total_written == 0
    assert rb.frames_available == 0
    assert rb.read(0, 100).size == 0


def test_concurrent_writers_do_not_corrupt_readers():
    """The capture thread writes while the analysis thread reads."""
    rb = RingBuffer(48000, 1.0)
    stop = threading.Event()
    errors: list[str] = []

    def writer():
        try:
            block = np.full(256, 0.5, dtype=np.float32)
            for _ in range(2000):
                if stop.is_set():
                    break
                rb.write(block)
        except Exception as exc:  # pragma: no cover
            errors.append(f"writer: {exc}")

    def reader():
        try:
            for _ in range(2000):
                if stop.is_set():
                    break
                rb.read_latest(512)
                rb.snapshot()
        except Exception as exc:  # pragma: no cover
            errors.append(f"reader: {exc}")

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    stop.set()

    assert not errors, errors
    assert rb.total_written > 0
    # Every retained sample must be the value that was written.
    snap = rb.snapshot()
    assert np.all(snap == 0.5)


def test_seconds_helpers():
    rb = RingBuffer(48000, 30.0)
    assert rb.frame_to_seconds(48_000) == 1.0
    assert rb.seconds_to_frame(2.0) == 96_000
    rb.write(np.zeros(96_000, dtype=np.float32))
    assert rb.time_of_latest_sample() == pytest.approx(2.0)
