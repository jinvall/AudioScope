"""Feature extraction, checked against signals with known answers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.analysis.features import (
    SILENCE_DB,
    FeatureExtractor,
    amplitude_to_db,
    db_to_amplitude,
)
from app.analysis.frames import ANALYSIS_BANDS

SR = 48000
N = 1920  # the configured frame length


@pytest.fixture
def extractor() -> FeatureExtractor:
    return FeatureExtractor(SR, window="hann")


def sine(freq: float, amplitude: float = 0.5, n: int = N, sr: int = SR):
    t = np.arange(n) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ----------------------------------------------------------------------
# Level features
# ----------------------------------------------------------------------
def test_rms_and_peak_of_a_sine(extractor):
    f = extractor.analyse(sine(1000.0, 0.5))
    assert f.rms == pytest.approx(0.5 / math.sqrt(2), rel=1e-3)
    assert f.peak == pytest.approx(0.5, rel=1e-3)


def test_crest_factor_of_a_sine_is_sqrt_two(extractor):
    f = extractor.analyse(sine(1000.0, 0.5))
    assert f.crest_factor == pytest.approx(math.sqrt(2), rel=1e-2)


def test_levels_are_measured_unwindowed(extractor):
    """A Hann window halves a sine's amplitude; levels must not be affected.

    This is the regression test for measuring level on the windowed frame.
    """
    f = extractor.analyse(sine(1000.0, 0.5))
    assert f.rms == pytest.approx(0.5 / math.sqrt(2), rel=1e-3)
    assert f.crest_factor == pytest.approx(math.sqrt(2), rel=1e-2)


def test_silence(extractor):
    f = extractor.analyse(np.zeros(N, dtype=np.float32))
    assert f.rms == 0.0
    assert f.peak == 0.0
    assert f.rms_db == SILENCE_DB
    assert f.crest_factor == 0.0
    assert f.zero_crossing_rate == 0.0
    assert f.spectral_centroid_hz == 0.0
    assert f.spectral_flatness == 0.0


def test_dc_offset_is_detected_and_ignored_by_zcr(extractor):
    f = extractor.analyse(np.full(N, 0.25, dtype=np.float32))
    assert f.dc_offset == pytest.approx(0.25, rel=1e-6)
    assert f.rms == pytest.approx(0.25, rel=1e-6)
    # A constant signal crosses zero zero times.
    assert f.zero_crossing_rate == 0.0


def test_zero_crossing_rate_matches_the_tone_frequency(extractor):
    for freq in (250.0, 1000.0, 4000.0):
        f = extractor.analyse(sine(freq))
        # A tone of frequency f spans f*N/sr complete cycles in the frame, so
        # it crosses zero 2*f*N/sr times, less the one lying on the frame
        # boundary.  The rate divides by the N-1 intervals between samples.
        crossings = 2 * freq * N / SR - 1
        assert f.zero_crossing_rate == pytest.approx(
            crossings / (N - 1), rel=0.01
        )


def test_crest_factor_is_high_for_an_impulse(extractor):
    impulse = np.zeros(N, dtype=np.float32)
    impulse[N // 2] = 1.0
    f = extractor.analyse(impulse)
    assert f.crest_factor > 10.0


# ----------------------------------------------------------------------
# Spectral features
# ----------------------------------------------------------------------
def test_spectral_centroid_tracks_the_tone(extractor):
    for freq in (100.0, 1000.0, 5000.0, 15000.0):
        f = extractor.analyse(sine(freq))
        assert f.spectral_centroid_hz == pytest.approx(freq, rel=0.02)


def test_rolloff_is_above_the_tone_and_below_nyquist(extractor):
    f = extractor.analyse(sine(1000.0))
    assert 1000.0 <= f.spectral_rolloff_hz < SR / 2


def test_flatness_distinguishes_tone_from_noise(extractor):
    tone = extractor.analyse(sine(1000.0))
    rng = np.random.default_rng(0)
    noise = extractor.analyse((0.1 * rng.standard_normal(N)).astype(np.float32))
    assert tone.spectral_flatness < 0.01
    assert noise.spectral_flatness > 0.2
    assert tone.spectral_flatness < noise.spectral_flatness


def test_bandwidth_is_narrow_for_a_tone_and_wide_for_noise(extractor):
    tone = extractor.analyse(sine(1000.0))
    rng = np.random.default_rng(1)
    noise = extractor.analyse((0.1 * rng.standard_normal(N)).astype(np.float32))
    assert tone.spectral_bandwidth_hz < noise.spectral_bandwidth_hz


def test_flatness_is_within_range(extractor):
    rng = np.random.default_rng(2)
    for amplitude in (0.001, 0.1, 0.9):
        f = extractor.analyse((amplitude * rng.standard_normal(N)).astype(np.float32))
        assert 0.0 <= f.spectral_flatness <= 1.0


# ----------------------------------------------------------------------
# Band energies
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "freq,expected_band",
    [
        (100.0, "80-250"),
        (300.0, "250-500"),
        (800.0, "500-1000"),
        (1500.0, "1000-2000"),
        (3000.0, "2000-4000"),
        (6000.0, "4000-8000"),
        (12000.0, "8000-16000"),
        (20000.0, "16000-24000"),
    ],
)
def test_a_tone_lands_in_the_right_band(extractor, freq, expected_band):
    f = extractor.analyse(sine(freq))
    loudest = max(f.bands, key=lambda b: b.linear)
    assert loudest.label == expected_band
    # Bands well away from the tone must be far below it.
    index = [b.label for b in f.bands].index(expected_band)
    distant = [b for i, b in enumerate(f.bands) if abs(i - index) >= 3]
    for band in distant:
        assert band.db < f.band_db(expected_band) - 40, band.label


def test_hann_leakage_into_an_adjacent_band_is_bounded(extractor):
    """Documented, measured limitation of a 40 ms frame.

    A Hann window's main lobe is four bins wide, so a tone near a band edge
    puts real energy into the neighbouring band: its window response is only
    about 6 dB down one bin from centre.  With 25 Hz bins at 40 ms this is
    unavoidable inside the documented 20-50 ms frame range.  The bound here is
    asserted so a change of window or frame length that made it worse would
    fail loudly rather than quietly distort a detector's input.
    """
    # 100 Hz sits one bin above the 80 Hz edge of the 20-80 band.
    f = extractor.analyse(sine(100.0))
    leak = f.band_db("20-80") - f.band_db("80-250")
    assert -12.0 < leak < 0.0, f"edge leakage {leak:.1f} dB"


def test_every_band_is_reported(extractor):
    f = extractor.analyse(sine(1000.0))
    assert len(f.bands) == len(ANALYSIS_BANDS)
    assert [b.label for b in f.bands] == [
        f"{lo:g}-{hi:g}" for lo, hi in ANALYSIS_BANDS
    ]


def test_quiet_bands_report_the_silence_floor(extractor):
    f = extractor.analyse(sine(1000.0))
    # A 1 kHz tone puts nothing in the top band.
    assert f.band_db("16000-24000") == pytest.approx(SILENCE_DB, abs=0.01)


def test_band_density_makes_bands_comparable(extractor):
    """Total band energy grows with bandwidth; density must not."""
    rng = np.random.default_rng(3)
    f = extractor.analyse((0.05 * rng.standard_normal(N)).astype(np.float32))
    densities = [b.density_db for b in f.bands]
    # For white noise the per-hertz density is flat to within a few dB.
    assert max(densities) - min(densities) < 12.0
    # While totals clearly do grow with width.
    totals = [b.db for b in f.bands]
    assert totals[-1] > totals[0] + 5.0


def test_band_lookup_by_label(extractor):
    f = extractor.analyse(sine(1000.0))
    assert f.band_dict["1000-2000"] == pytest.approx(f.band_db("1000-2000"))
    with pytest.raises(KeyError):
        f.band_db("no-such-band")


# ----------------------------------------------------------------------
# Spectral flux
# ----------------------------------------------------------------------
def test_flux_is_absent_on_the_first_frame(extractor):
    """Honest absence, not a fabricated zero."""
    assert extractor.analyse(sine(1000.0)).spectral_flux is None


def test_flux_is_nonzero_when_the_spectrum_changes(extractor):
    extractor.analyse(sine(1000.0))
    f = extractor.analyse(sine(6000.0))
    assert f.spectral_flux is not None
    assert f.spectral_flux > 0.0


def test_flux_is_near_zero_for_a_steady_tone(extractor):
    extractor.analyse(sine(1000.0))
    f = extractor.analyse(sine(1000.0))
    assert f.spectral_flux is not None
    assert f.spectral_flux < 0.05


def test_flux_resets_with_the_extractor(extractor):
    extractor.analyse(sine(1000.0))
    extractor.reset()
    assert extractor.analyse(sine(1000.0)).spectral_flux is None


# ----------------------------------------------------------------------
# Plumbing
# ----------------------------------------------------------------------
def test_db_conversion_round_trip():
    assert amplitude_to_db(1.0) == pytest.approx(0.0, abs=1e-9)
    assert amplitude_to_db(0.0) == pytest.approx(SILENCE_DB)
    assert amplitude_to_db(db_to_amplitude(-20.0)) == pytest.approx(-20.0)
    assert amplitude_to_db(-1.0) == pytest.approx(0.0, abs=1e-9)


def test_frequencies_property_requires_a_frame(extractor):
    with pytest.raises(RuntimeError):
        extractor.frequencies
    extractor.analyse(sine(1000.0))
    freqs = extractor.frequencies
    assert freqs.size == N // 2 + 1
    assert freqs[-1] == pytest.approx(SR / 2, rel=0.01)


def test_empty_frame_is_rejected(extractor):
    with pytest.raises(ValueError):
        extractor.analyse(np.zeros(0, dtype=np.float32))


def test_bad_construction_is_rejected():
    with pytest.raises(ValueError):
        FeatureExtractor(0)
    with pytest.raises(ValueError):
        FeatureExtractor(SR, rolloff_percentile=0.0)
    with pytest.raises(ValueError):
        FeatureExtractor(SR, rolloff_percentile=1.5)


def test_frame_index_advances(extractor):
    for expected in range(5):
        assert extractor.analyse(sine(1000.0)).index == expected


def test_start_sample_is_recorded(extractor):
    assert extractor.analyse(sine(1000.0), start_sample=4800).start_sample == 4800


def test_features_serialise(extractor):
    import json

    payload = extractor.analyse(sine(1000.0)).to_dict()
    text = json.dumps(payload)
    assert "spectral_centroid_hz" in text
    assert len(payload["bands"]) == len(ANALYSIS_BANDS)


def test_no_feature_is_nan_for_odd_input(extractor):
    rng = np.random.default_rng(4)
    for data in (
        np.zeros(N, dtype=np.float32),
        np.full(N, 1e-9, dtype=np.float32),
        (rng.standard_normal(N) * 1e-6).astype(np.float32),
        np.ones(N, dtype=np.float32),
    ):
        f = extractor.analyse(data)
        for name in (
            "rms", "peak", "crest_factor", "zero_crossing_rate", "dc_offset",
            "spectral_centroid_hz", "spectral_bandwidth_hz",
            "spectral_rolloff_hz", "spectral_flatness",
        ):
            value = getattr(f, name)
            assert not math.isnan(value), f"{name} is NaN"
        for band in f.bands:
            assert not math.isnan(band.db)
