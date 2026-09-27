"""Frame extraction (docs/AUDIO_PIPELINE.md section 4)."""

from __future__ import annotations

import numpy as np
import pytest

from app.analysis.frames import (
    ANALYSIS_BANDS,
    BAND_LABELS,
    FrameExtractor,
    window_function,
)


def test_band_set_matches_the_specification():
    """AGENTS.md section 7 lists the bands explicitly."""
    assert ANALYSIS_BANDS == (
        (20.0, 80.0),
        (80.0, 250.0),
        (250.0, 500.0),
        (500.0, 1000.0),
        (1000.0, 2000.0),
        (2000.0, 4000.0),
        (4000.0, 8000.0),
        (8000.0, 16000.0),
        (16000.0, 24000.0),
    )
    assert len(BAND_LABELS) == len(ANALYSIS_BANDS)


def test_frames_land_on_the_hop_grid():
    extractor = FrameExtractor(100, 40, None)
    frames = extractor.push(np.arange(500, dtype=np.float32))
    # Starts at 0, 40, 80, ... while a whole frame still fits.
    assert frames[0][0] == 0.0
    assert frames[1][0] == 40.0
    assert frames[2][0] == 80.0


@pytest.mark.parametrize("block", [1, 7, 64, 100, 333, 1024, 4096])
def test_result_is_independent_of_block_size(block):
    """A different device block size must not move the frames.

    Otherwise every feature value would depend on the device's buffering.
    """
    data = np.arange(20000, dtype=np.float32)
    reference = FrameExtractor(480, 160, None)
    expected = []
    for i in range(0, data.size, 997):
        expected += reference.push(data[i:i + 997])

    other = FrameExtractor(480, 160, None)
    actual = []
    for i in range(0, data.size, block):
        actual += other.push(data[i:i + block])

    assert len(actual) == len(expected)
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected))


def test_no_audio_is_skipped_between_frames():
    """Each frame must equal its own slice of the input."""
    data = np.arange(10000, dtype=np.float32)
    extractor = FrameExtractor(500, 250, None)
    frames = []
    for i in range(0, data.size, 321):
        frames += extractor.push(data[i:i + 321])
    for index, frame in enumerate(frames):
        start = index * 250
        assert np.array_equal(frame, data[start:start + 500])


def test_partial_trailing_data_is_buffered_not_emitted():
    extractor = FrameExtractor(1000, 100, None)
    frames = extractor.push(np.zeros(500, dtype=np.float32))
    assert frames == []
    assert extractor.buffered_samples == 500
    frames = extractor.push(np.zeros(500, dtype=np.float32))
    # One frame (starting at 0) is now complete; the grid has advanced by the
    # hop, so the remainder is everything after that frame's start.
    assert len(frames) == 1
    assert extractor.buffered_samples == 900


def test_memory_stays_bounded_over_a_long_run():
    """docs/PERFORMANCE.md section 7: no unbounded growth."""
    extractor = FrameExtractor(1920, 480, None)
    for _ in range(2000):
        extractor.push(np.zeros(1024, dtype=np.float32))
    # Bounded by one frame plus at most one block.
    assert extractor.buffered_samples <= 1920 + 1024


def test_window_is_applied_when_requested():
    extractor = FrameExtractor(4, 1, "hann")
    frames = extractor.push(np.ones(4, dtype=np.float32))
    assert frames[0].shape == (4,)
    # A Hann window is zero at the first sample and below 1 elsewhere.
    assert frames[0][0] < 0.01
    assert 0.0 < frames[0][1] < 1.0


def test_no_window_by_default():
    extractor = FrameExtractor(4, 1, None)
    frames = extractor.push(np.ones(4, dtype=np.float32))
    assert np.array_equal(frames[0], np.ones(4, dtype=np.float32))


def test_flush_discards_an_incomplete_tail_rather_than_padding():
    """Padding with zeros would teach the noise floor that the room is silent."""
    extractor = FrameExtractor(100, 50, None)
    extractor.push(np.ones(150, dtype=np.float32))
    assert extractor.buffered_samples > 0
    assert extractor.flush() == []
    assert extractor.tail_samples_unanalysed > 0
    assert extractor.padded_frames == 0


def test_flush_can_pad_when_explicitly_requested():
    extractor = FrameExtractor(100, 50, None)
    extractor.push(np.ones(150, dtype=np.float32))
    tails = extractor.flush(pad=True)
    assert len(tails) == 1
    assert tails[0].size == 100
    # Only the part with real audio is non-zero; the rest is declared padding.
    assert 0 < np.count_nonzero(tails[0]) < 100
    assert extractor.padded_frames == 1


def test_flush_with_nothing_buffered():
    extractor = FrameExtractor(100, 50, None)
    assert extractor.flush() == []
    assert extractor.flush(pad=True) == []


def test_reset_clears_state():
    extractor = FrameExtractor(100, 50, None)
    extractor.push(np.ones(500, dtype=np.float32))
    extractor.reset()
    assert extractor.buffered_samples == 0
    assert extractor.frames_emitted == 0
    # After a reset the grid restarts, so the first frame is the first samples.
    frames = extractor.push(np.arange(200, dtype=np.float32))
    assert frames[0][0] == 0.0


def test_empty_input_is_a_no_op():
    extractor = FrameExtractor(100, 50, None)
    assert extractor.push(np.zeros(0, dtype=np.float32)) == []


@pytest.mark.parametrize("frame,hop", [(0, 10), (100, 0), (100, 200), (-1, 10)])
def test_invalid_geometry_is_rejected(frame, hop):
    with pytest.raises(ValueError):
        FrameExtractor(frame, hop, None)


def test_unknown_window_is_rejected():
    with pytest.raises(ValueError):
        window_function("triangle", 8)


@pytest.mark.parametrize("name", ["hann", "hamming", "blackman", "rect", "none"])
def test_windows_have_the_requested_length(name):
    assert window_function(name, 16).size == 16
    assert window_function(name, 1).size == 1
    assert window_function(name, 0).size == 0
