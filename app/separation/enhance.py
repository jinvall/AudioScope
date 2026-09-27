"""Optional, deliberately conservative enhancement of a separated signal.

AGENTS.md section 17 is the constraint that shapes this module: enhancement
is optional, and it may only remove DC, apply conservative normalisation,
prevent clipping, and clean up gently.  Aggressive processing is forbidden
because the separated signal *is* the evidence being investigated - a
denoiser that also removes the thing being listened for is worse than no
enhancement at all.

Every function here returns a new array.  The raw model output
(``isolated.wav``) is never touched, and enhancement is written to a separate
``enhanced.wav`` (docs/SOURCE_SEPARATION.md section 11).
"""

from __future__ import annotations

import numpy as np

from ..audio.wavio import write_wav_atomic


def remove_dc(samples: np.ndarray) -> np.ndarray:
    """Subtract the mean.

    DC offset in a separated track is an artefact of the separation, not part
    of the isolated sound, and it wastes headroom.
    """
    if samples.size == 0:
        return samples
    return samples - float(np.mean(samples))


def normalise_peak(
    samples: np.ndarray,
    target_peak_dbfs: float = -3.0,
) -> tuple[np.ndarray, float]:
    """Scale so the peak sits at ``target_peak_dbfs``.  Returns (audio, gain).

    A gain of exactly 1.0 means no change was needed.  Silence is left
    alone: normalising digital silence would manufacture a signal out of
    nothing, and every subsequent measurement would then describe the gain
    rather than the audio.
    """
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 0.0:
        return samples, 1.0
    target = 10.0 ** (target_peak_dbfs / 20.0)
    gain = target / peak
    # 1.0 is not rounded away: a gain below 1 is a real, if small, change and
    # belongs in the metadata.
    return samples * gain, float(gain)


def soft_clip(samples: np.ndarray, ceiling: float = 0.999) -> np.ndarray:
    """Bound the signal without the flat-topping of hard clipping.

    Only ever applied *after* normalisation has targeted a peak below
    full scale, so in normal operation it is a no-op.  It exists so that a
    pathological input cannot produce a file that damages playback.
    """
    limit = min(ceiling, 1.0)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= limit:
        return samples
    # tanh(x/k) scaled to reach `limit` at x = peak.
    k = peak / np.arctanh(limit / peak)
    return limit * np.tanh(samples / k)


def enhance(
    samples: np.ndarray,
    target_peak_dbfs: float = -3.0,
    remove_dc_offset: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply the permitted operations in a fixed, documented order.

    DC removal first (it changes the mean, and therefore the peak), then
    normalisation, then the clip guard.  The applied gain is returned so it
    can be recorded; a listener comparing the raw and enhanced files should be
    able to see exactly what changed.
    """
    applied: dict[str, float] = {}
    out = np.asarray(samples, dtype=np.float32).copy()

    if remove_dc_offset and out.size:
        before = float(np.mean(out))
        out = remove_dc(out)
        applied["dc_removed"] = before

    out, gain = normalise_peak(out, target_peak_dbfs)
    applied["gain"] = gain
    out = soft_clip(out)

    applied["peak"] = float(np.max(np.abs(out))) if out.size else 0.0
    return out, applied


def enhance_file(
    isolated_path: str,
    enhanced_path: str,
    target_peak_dbfs: float = -3.0,
    remove_dc_offset: bool = True,
) -> dict[str, float]:
    """Write ``enhanced.wav`` beside an existing ``isolated.wav``.

    ``isolated_path`` is opened read-only.  The separation is the evidence;
    enhancement is a convenience copy.
    """
    from ..audio.wavio import read_wav

    raw = read_wav(isolated_path)
    enhanced, applied = enhance(
        raw, target_peak_dbfs, remove_dc_offset
    )
    # soundfile names it `samplerate`; `sample_rate` is this project's own
    # WavInfo spelling, and mixing the two is an AttributeError.
    info = _info(isolated_path)
    write_wav_atomic(enhanced_path, enhanced, int(info.samplerate))
    return applied


def _info(path: str) -> object:
    import soundfile as sf

    return sf.info(path)
