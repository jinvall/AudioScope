"""Resampler: block-independence and conversion quality."""

from __future__ import annotations

import numpy as np
import pytest

from app.audio.resample import (
    StreamingResampler,
    _gcd,
    design_prototype,
    ratio_for,
)


def test_ratio_for_reduces_to_coprime():
    assert ratio_for(44100, 48000) == (160, 147)
    assert ratio_for(48000, 44100) == (147, 160)
    assert ratio_for(48000, 48000) == (1, 1)
    # Already coprime.
    assert ratio_for(8000, 22050) == (441, 160)


def test_gcd():
    assert _gcd(480, 160) == 160
    assert _gcd(147, 160) == 1


def test_identity_is_passthrough():
    r = StreamingResampler(1, 1)
    assert r.is_identity
    x = np.arange(10, dtype=np.float32)
    assert np.array_equal(r.process(x), x)


def test_output_length_is_exact():
    """Total output must be exactly floor(n * up / down)."""
    n = 44100
    x = np.zeros(n, dtype=np.float32)
    up, down = ratio_for(44100, 48000)
    r = StreamingResampler(up, down)
    out = r.process(x)
    assert out.size == (n * up) // down


@pytest.mark.parametrize("block", [1, 147, 512, 777, 1024, 4096, 8192])
def test_result_is_independent_of_block_size(block):
    """The property the streaming implementation exists to guarantee.

    Concatenating the output of many small calls must equal the output of a
    single call, sample for sample.  If this fails, block boundaries click.
    """
    rate = 44100
    n = 44100
    t = np.arange(n) / rate
    x = (
        0.4 * np.sin(2 * np.pi * 1000 * t)
        + 0.2 * np.sin(2 * np.pi * 300 * t)
        + 0.01 * np.random.default_rng(0).standard_normal(n)
    ).astype(np.float32)

    up, down = ratio_for(44100, 48000)

    whole = StreamingResampler(up, down).process(x)

    chunked_r = StreamingResampler(up, down)
    parts = [
        chunked_r.process(x[i:i + block]) for i in range(0, n, block)
    ]
    chunked = np.concatenate([p for p in parts if p.size])

    assert chunked.size == whole.size
    assert np.array_equal(chunked, whole), (
        f"block size {block}: max deviation "
        f"{np.max(np.abs(chunked - whole))}"
    )


def test_output_length_independent_of_block_size():
    rate, n = 44100, 44100
    x = np.zeros(n, dtype=np.float32)
    up, down = ratio_for(44100, 48000)
    for block in (147, 1024, 4096):
        r = StreamingResampler(up, down)
        parts = [r.process(x[i:i + block]) for i in range(0, n, block)]
        total = sum(p.size for p in parts)
        assert total == (n * up) // down, block


@pytest.mark.parametrize("freq,expected", [(300, 0.2), (1000, 0.4), (8000, 0.4)])
def test_passband_amplitude_is_preserved(freq, expected):
    """A tone in the passband must come out at its input amplitude."""
    rate, n = 44100, 44100
    t = np.arange(n) / rate
    x = (expected * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    up, down = ratio_for(44100, 48000)
    y = StreamingResampler(up, down).process(x)

    out_t = np.arange(y.size) / 48000
    amplitude = abs(2 * np.mean(y * np.exp(-2j * np.pi * freq * out_t)))
    assert amplitude == pytest.approx(expected, rel=0.02)


def test_matches_scipy_resample_poly_response():
    """Our filter must have the same frequency response as SciPy's.

    Guards against a prototype that is subtly wrong (wrong cutoff, wrong
    normalisation) while still allowing the documented constant delay.
    """
    from scipy.signal import resample_poly

    rate, n = 44100, 22050
    t = np.arange(n) / rate
    up, down = ratio_for(44100, 48000)

    for freq in (500.0, 4000.0, 12000.0):
        x = np.sin(2 * np.pi * freq * t).astype(np.float32)
        ours = StreamingResampler(up, down).process(x)
        theirs = resample_poly(x, up, down).astype(np.float32)

        k = min(ours.size, theirs.size)
        ours_t = np.arange(k) / 48000
        a_ours = abs(2 * np.mean(ours[:k] * np.exp(-2j * np.pi * freq * ours_t)))
        a_theirs = abs(
            2 * np.mean(theirs[:k] * np.exp(-2j * np.pi * freq * ours_t))
        )
        assert a_ours == pytest.approx(a_theirs, rel=0.02), freq


def test_dc_is_preserved():
    """Constant input must give constant output at the same level."""
    x = np.full(44100, 0.25, dtype=np.float32)
    up, down = ratio_for(44100, 48000)
    y = StreamingResampler(up, down).process(x)
    # Ignore the filter's start-up transient.
    assert np.mean(y[5000:]) == pytest.approx(0.25, rel=1e-3)


def test_prototype_dc_gain_matches_upsampling_factor():
    up, down = ratio_for(44100, 48000)
    h = design_prototype(up, down)
    assert h.sum() == pytest.approx(up, rel=1e-6)


def test_upsampling_and_downsampling_round_trip():
    """44.1k -> 48k -> 44.1k should return to the original signal.

    Each conversion adds its FIR group delay, so the round trip is compared
    after aligning by the delay the filters actually introduced.  Comparing
    the raw arrays would only measure the delay, not the conversion quality.
    """
    rate, n = 44100, 22050
    t = np.arange(n) / rate
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    up_once = StreamingResampler(*ratio_for(44100, 48000)).process(x)
    back = StreamingResampler(*ratio_for(48000, 44100)).process(up_once)

    assert back.size == pytest.approx(n, rel=0.01)

    # Find the delay by cross-correlating against the original.  The window is
    # kept shorter than one cycle of the test tone (100 samples at 440 Hz /
    # 44.1 kHz) so the correlation peak is unique; with a longer window a sine
    # correlates equally well at every cycle offset and the argmax is
    # meaningless.
    window = 50
    reference = x[2000:2000 + window].astype(np.float64)
    measured = back.astype(np.float64)
    correlation = np.correlate(measured, reference, mode="full")
    peak = int(np.argmax(correlation))
    # In 'full' mode the correlation peak at index `peak` places reference[0]
    # at measured index `peak - (window - 1)`.
    lag = peak - (window - 1) - 2000

    start = 2000 + lag
    stop = min(2000 + lag + 4000, back.size, n)
    assert stop - start > 1000, f"implausible delay {lag}"
    residual = np.max(np.abs(back[start:stop] - x[2000:2000 + (stop - start)]))
    assert residual < 0.05, f"residual {residual} after lag {lag}"


def test_empty_block_is_safe():
    r = StreamingResampler(*ratio_for(44100, 48000))
    assert r.process(np.zeros(0, np.float32)).size == 0


def test_reset_clears_state():
    up, down = ratio_for(44100, 48000)
    x = np.sin(np.arange(5000) * 0.1).astype(np.float32)
    first = StreamingResampler(up, down).process(x)
    r = StreamingResampler(up, down)
    r.process(x)
    r.reset()
    second = r.process(x)
    assert np.array_equal(first, second)


def test_rejects_bad_ratios():
    with pytest.raises(ValueError):
        StreamingResampler(0, 147)
    with pytest.raises(ValueError):
        StreamingResampler(160, -1)
    with pytest.raises(ValueError):
        ratio_for(0, 48000)
