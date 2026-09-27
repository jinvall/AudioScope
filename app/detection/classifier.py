"""Event classification.

docs/DETECTION_AND_CLASSIFICATION.md. The rules below answer "what kind of
acoustic event might this be?", and they are allowed to answer *unknown*:
section 7 says unknown is valid and the classifier must not force a label.

Confidence
----------
**This module reports ``confidence = None``, always.**

That is not an omission.  docs/DETECTION_AND_CLASSIFICATION.md section 8
requires confidence to come from an actual calibrated model output and forbids
generating it from "arbitrary cosmetic scaling".  What follows *is* a classifier,
and its output score is reported as ``label_scores`` - a real, reproducible
number computed from the measurements.  But a rule-based score is not a
calibrated probability, so presenting it as a confidence percentage would be
exactly the cosmetic number the specification prohibits.  A UI should show the
label, the evidence, the competing label and the margin.

Evidence
--------
Every rule records the observations that fired, with the measured values, so a
label can be argued with rather than merely believed.  Section 9 of the
specification asks for exactly this shape:

    classification: possible_whisper
    evidence:       low_rms, elevated_high_frequency_energy, ...

Threshold provenance
--------------------
Thresholds marked "measured" come from running signals of a known kind through
this pipeline and reading the distributions, not from guesswork.  Where a
threshold is a judgement call, it says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..config import DetectionConfig
from .tracker import EventProfile

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
UNKNOWN = "unknown"

LABEL_WHISPER = "possible_whisper"
LABEL_SPEECH = "possible_speech"
LABEL_FOOTSTEP = "possible_footstep"
LABEL_KNOCK = "possible_knock"
LABEL_MOVEMENT = "possible_movement"
LABEL_CLOTHING = "possible_clothing"
LABEL_SCRAPING = "possible_scraping"
LABEL_INTERFERENCE = "possible_interference"
LABEL_CLIPPING = "possible_clipping"

#: Separation queries suggested per label (docs section 10).  The user can always
#: override these.
SEPARATION_QUERIES = {
    LABEL_WHISPER: "a whisper",
    LABEL_SPEECH: "speech",
    LABEL_FOOTSTEP: "footsteps",
    LABEL_KNOCK: "a knock",
    LABEL_MOVEMENT: "a person moving",
    LABEL_CLOTHING: "clothing rustling",
    LABEL_SCRAPING: "a scraping sound",
    LABEL_INTERFERENCE: "electrical interference",
    LABEL_CLIPPING: "a clicking sound",
    UNKNOWN: "a sound",
}

# Confidence is deliberately absent. See the module docstring.
CONFIDENCE_NOTE = (
    "Rule-based classifier. Scores are reproducible measurements, not "
    "calibrated probabilities, so no confidence percentage is reported "
    "(docs/DETECTION_AND_CLASSIFICATION.md section 8)."
)


@dataclass
class LabelScore:
    """One label's score, with the evidence that produced it."""

    label: str
    score: float
    evidence: tuple[str, ...] = ()
    details: dict[str, float] = field(default_factory=dict)


@dataclass
class Classification:
    """The classifier's verdict for one event."""

    classification: str
    confidence: Optional[float] = None
    confidence_note: str = CONFIDENCE_NOTE
    evidence: tuple[str, ...] = ()
    label_scores: dict[str, float] = field(default_factory=dict)
    # Per-label measured detail behind each score.  Kept alongside the scores
    # deliberately: a future false positive has to be explainable from measured
    # features rather than guessed thresholds.
    label_details: dict[str, dict] = field(default_factory=dict)
    runner_up: Optional[str] = None
    margin: float = 0.0
    separation_query: str = SEPARATION_QUERIES[UNKNOWN]
    ambiguous: bool = False

    def to_dict(self) -> dict:
        return {
            "classification": self.classification,
            "confidence": self.confidence,
            "confidence_note": self.confidence_note,
            "evidence": list(self.evidence),
            "label_scores": dict(self.label_scores),
            "label_details": {
                k: dict(v) for k, v in self.label_details.items()
            },
            "runner_up": self.runner_up,
            "margin": self.margin,
            "separation_query": self.separation_query,
            "ambiguous": self.ambiguous,
        }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _ramp(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    if value != value:  # NaN
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _harmonic_dominant(hz: float) -> bool:
    """True for 50/60 Hz mains or a low harmonic of it.

    The tolerance is deliberately tight, at 6%.  With a 40 ms frame the FFT
    bins are 25 Hz apart at 48 kHz, so a looser tolerance makes a 90 Hz footstep
    thump look like 100 Hz mains - which is exactly the false positive this
    function was tightened to remove.  A real 50 Hz hum lands on bin 2.0
    exactly, so a tight tolerance costs nothing on genuine interference.
    """
    if hz <= 0.0:
        return False
    for base in (50.0, 60.0):
        for multiple in (1, 2, 3, 4, 5):
            target = base * multiple
            if abs(hz - target) <= 0.06 * target:
                return True
    return False


def _repetition_score(intervals: list[float]) -> tuple[float, float]:
    """Score a sequence of onsets as a plausible repeated-event cadence.

    docs/DETECTION_AND_CLASSIFICATION.md section 4: several events in sequence
    are stronger evidence than one, but a *fixed* cadence must not be required.
    So this rewards a number of onsets with reasonably regular spacing, and
    returns the regularity alongside.

    Returns ``(score, regularity)`` where regularity is 0 for a single onset and
    1 for a perfectly even sequence.
    """
    if len(intervals) < 1:
        return 0.0, 0.0
    # Onset count. Measured need: three evenly spaced impacts is already a
    # walking sequence and must clear the 0.4 "is this a sequence" bar that the
    # footstep rule uses to stop penalising a sequence for its length. A ramp
    # that saturated at six onsets left a normal three-step sequence scoring
    # 0.25, under the bar.
    count = _ramp(len(intervals) + 1, 1.0, 4.0)
    if len(intervals) < 2:
        return 0.5 * count, 0.0
    values = np.asarray(intervals, dtype=np.float64)
    # Only intervals in a human-plausible range count as a sequence.
    plausible = values[(values >= 0.15) & (values <= 2.5)]
    if plausible.size < 2:
        return 0.3 * count, 0.0
    mean = float(plausible.mean())
    spread = float(plausible.std())
    regularity = float(np.exp(-spread / mean)) if mean > 0 else 0.0
    return min(1.0, count * (0.60 + 0.40 * regularity)), regularity


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------
class Classifier:
    """Scores an event against each label's evidence rules."""

    def __init__(
        self,
        config: Optional[DetectionConfig] = None,
        whisper_rms_dbfs: float = -42.0,
        speech_rms_dbfs: float = -50.0,
        min_label_score: float = 0.18,
        whisper_min_score: float = 0.22,
        ambiguity_margin: float = 0.08,
    ) -> None:
        from ..config import DetectionConfig

        self.config = config or DetectionConfig()
        self.whisper_min_score = whisper_min_score
        # "Low RMS" for a whisper, and a floor below which something is simply
        # too quiet to call.  Measured: the pipeline's silence floor sits at
        # -200 dBFS and ordinary room noise in a quiet room is around -54, so
        # -42 dBFS is well above noise yet clearly below a speaking voice.
        self.whisper_rms_dbfs = whisper_rms_dbfs
        self.speech_rms_dbfs = speech_rms_dbfs
        # Below this, no rule cleared its own bar and the answer is unknown
        # (docs section 7: never force a classification).
        self.min_label_score = min_label_score
        # If the top two labels are within this, the label is flagged ambiguous.
        self.ambiguity_margin = ambiguity_margin

    # ------------------------------------------------------------------
    def classify(
        self, profile: EventProfile, duration: float
    ) -> Classification:
        scores: list[LabelScore] = []
        for rule in (
            self._rule_clipping,
            self._rule_interference,
            self._rule_whisper,
            self._rule_speech,
            self._rule_footstep,
            self._rule_knock,
            self._rule_scraping,
            self._rule_clothing,
            self._rule_movement,
        ):
            result = rule(profile, duration)
            if result is not None and result.score > 0.0:
                scores.append(result)

        if not scores:
            return Classification(
                classification=UNKNOWN,
                evidence=("no_rule_matched",),
                separation_query=SEPARATION_QUERIES[UNKNOWN],
            )

        scores.sort(key=lambda s: s.score, reverse=True)
        best = scores[0]
        second = scores[1] if len(scores) > 1 else None
        margin = best.score - (second.score if second else 0.0)

        if best.score < self.min_label_score:
            # Evidence was present but weak. Saying so is more useful than
            # picking the least-bad label.
            return Classification(
                classification=UNKNOWN,
                evidence=tuple(best.evidence) + ("weak_evidence",),
                label_scores={s.label: s.score for s in scores},
                label_details={s.label: dict(s.details) for s in scores},
                runner_up=best.label,
                margin=best.score,
                separation_query=SEPARATION_QUERIES[UNKNOWN],
                ambiguous=True,
            )

        return Classification(
            classification=best.label,
            evidence=best.evidence,
            label_scores={s.label: s.score for s in scores},
            label_details={s.label: dict(s.details) for s in scores},
            runner_up=second.label if second else None,
            margin=margin,
            separation_query=SEPARATION_QUERIES.get(
                best.label, SEPARATION_QUERIES[UNKNOWN]
            ),
            ambiguous=second is not None and margin < self.ambiguity_margin,
        )

    # ------------------------------------------------------------------
    # Each rule returns None or a LabelScore with score in 0..1.
    # ------------------------------------------------------------------
    def _rule_clipping(self, p: EventProfile, duration: float):
        if p.max_clipping_ratio <= 0.0:
            return None
        severity = _ramp(p.max_clipping_ratio, 0.001, 0.03)
        return LabelScore(
            label=LABEL_CLIPPING,
            score=min(1.0, 0.7 + 0.3 * severity),
            evidence=("input_clipping", "clipped_samples_present"),
            details={"max_clipping_ratio": p.max_clipping_ratio},
        )

    def _rule_interference(self, p: EventProfile, duration: float):
        """Mains hum and electrical buzz.

        Tonal and sitting on 50/60 Hz or a harmonic of it.  Requires both: a
        tone at 800 Hz is a motor, not electrical interference.
        """
        # A hum is a genuinely narrow tone.  Measured: broadband noise scores
        # about 13 on peakiness, a harmonic stack about 213, a pure tone 540-640.
        # Requiring real tonality stops a decaying thump from being called
        # interference just because one of its harmonics lands near a mains
        # frequency.
        tonal = _ramp(p.max_peakiness, 60.0, 150.0)
        # Narrow: the energy occupies a small fraction of the spectrum.
        narrow = _narrowness(p)
        mains = 1.0 if _harmonic_dominant(p.dominant_frequency_hz) else 0.0
        # Interference is steady.  An event with repeated sharp onsets is a
        # sequence of impacts, whatever its pitch, so this is a gate rather than
        # a small term: with 25 Hz bins a decaying thump and a mains harmonic
        # can share a bin, and steadiness is the property that separates them.
        if p.impulsive_onsets >= 2:
            return None
        score = tonal * (0.45 + 0.40 * mains + 0.15 * narrow)
        if score < 0.20:
            return None
        evidence = ["tonal_spectrum"]
        details = {
            "max_peakiness": p.max_peakiness,
            "dominant_frequency_hz": p.dominant_frequency_hz,
            "narrowness": _narrowness(p),
        }
        if mains:
            evidence.append("mains_frequency_or_harmonic")
            details["mains_harmonic"] = 1.0
        if narrow > 0.4:
            evidence.append("narrow_spectrum")
        return LabelScore(
            label=LABEL_INTERFERENCE, score=score, evidence=tuple(evidence),
            details=details,
        )

    def _rule_whisper(self, p: EventProfile, duration: float):
        """A whisper: quiet, sustained, speech-like, broadband.

        The rule is:

            WHISPER = quiet + sustained presence + speech-like temporal structure

        not "low mean RMS".  The distinction matters because a sequence of
        footsteps has a *low mean RMS for the same reason a whisper does* - the
        gaps between steps pull the event average down - and a mean spectral
        flatness that looks broadband for the same reason.  So:

        * the level used is :attr:`EventProfile.active_level_dbfs`, the median
          of frames that were actually above the adaptive floor, not the event
          mean;
        * the flatness used is :attr:`EventProfile.active_flatness`, averaged
          over those same frames, so gap frames cannot make an impulsive event
          look broadband;
        * presence is a *multiplicative* gate, because a sound that is only
          present a tenth of the time is not a sustained whisper however quiet
          it is between bursts;
        * impulsive onsets are a graded penalty, never an exclusion, because
          whispered speech contains consonant transients and must stay
          classifiable.
        """
        cfg = self.config

        # --- quiet, measured only where the sound is present -------------
        # A loud event must score zero here.  The ramp runs *downward* with
        # level: inverting this once made every loud event look like a whisper.
        quiet = 1.0 - _ramp(
            p.active_level_dbfs, self.whisper_rms_dbfs, self.whisper_rms_dbfs + 25.0
        )

        # --- broadband, from above-floor frames only ---------------------
        broadband = _ramp(p.active_flatness, 0.02, 0.25)

        # --- more energy up high than down low -----------------------------
        high_lean = _ramp(p.high_to_low_ratio_db, -2.0, 8.0)

        # --- present above the noise floor, so not silence -----------------
        above_floor = _ramp(p.peak_snr_db, 6.0, 15.0)

        # --- speech-like temporal structure --------------------------------
        modulated = _ramp(p.mean_modulation, cfg.modulation_low,
                          cfg.modulation_full)

        # --- weak harmonic structure ---------------------------------------
        # Peakiness of above-floor frames, so gap frames cannot make a tonal
        # event look harmonic-free.
        not_harmonic = 1.0 - _ramp(p.peakiness_above_max, 20.0, 90.0)
        no_fundamental = _ramp(p.active_centroid_hz, 600.0, 1800.0)

        # --- sustained presence: a multiplicative gate ---------------------
        present = _ramp(
            p.presence_fraction,
            cfg.whisper_presence_low,
            cfg.whisper_presence_full,
        )
        presence_gate = (
            cfg.whisper_presence_floor_weight
            + (1.0 - cfg.whisper_presence_floor_weight) * present
        )

        # --- impulsive onsets: a graded penalty, never an exclusion --------
        # Whispered speech has consonant transients.  A sequence dominated by
        # isolated strong transients should score poorly, but a whisper with a
        # few transients in it must remain classifiable, so this subtracts a
        # bounded amount rather than returning None.
        whisper_impulse_penalty = cfg.whisper_impulse_penalty * _ramp(
            p.impulse_density,
            cfg.whisper_impulse_density_low,
            cfg.whisper_impulse_density_full,
        )

        base = (
            0.24 * quiet
            + 0.16 * broadband
            + 0.20 * high_lean
            + 0.14 * above_floor
            + 0.14 * modulated
            + 0.12 * not_harmonic
        )
        score = base * presence_gate - whisper_impulse_penalty

        if score < self.whisper_min_score:
            return LabelScore(
                label=LABEL_WHISPER,
                score=max(0.0, score),
                evidence=("whisper_evidence_weak",),
                details=self._whisper_diagnostics(
                    p, quiet, broadband, high_lean, above_floor, modulated,
                    not_harmonic, present, presence_gate,
                    whisper_impulse_penalty, score,
                ),
            )

        evidence = []
        if quiet > 0.4:
            evidence.append("low_rms")
        if broadband > 0.4:
            evidence.append("broadband_no_strong_harmonics")
        if high_lean > 0.4:
            evidence.append("elevated_high_frequency_energy")
        if above_floor > 0.4:
            evidence.append("above_noise_floor_not_silence")
        if modulated > 0.4:
            evidence.append("speech_like_modulation")
        if present > 0.4:
            evidence.append("sustained_presence")
        if whisper_impulse_penalty > 0.15:
            evidence.append("impulsive_onsets_penalised")
        if not_harmonic > 0.4:
            evidence.append("weak_fundamental")

        return LabelScore(
            label=LABEL_WHISPER,
            score=score,
            evidence=tuple(evidence),
            details=self._whisper_diagnostics(
                p, quiet, broadband, high_lean, above_floor, modulated,
                not_harmonic, present, presence_gate,
                whisper_impulse_penalty, score,
            ),
        )

    def _family_onset_gate(self, p: EventProfile) -> float:
        """Scale the movement family down when the event is all sharp onsets.

        Movement, clothing and scraping are diffuse sounds.  A sequence of
        distinct impulses is a footstep sequence or a series of knocks, so those
        labels are suppressed in proportion to the onset density.  This is the
        mirror image of the whisper rule and is what stops a footstep sequence
        from being read as cloth rustle.
        """
        cfg = self.config
        return 1.0 - cfg.family_onset_suppression * _ramp(
            p.impulse_density,
            cfg.family_onset_density_low,
            cfg.family_onset_density_full,
        )

    def _whisper_diagnostics(self, p, *terms) -> dict:
        """Explainable evidence for a whisper decision (see the spec)."""
        (
            quiet, broadband, high_lean, above_floor, modulated,
            not_harmonic, present, presence_gate, penalty, score,
        ) = terms
        return {
            "whisper_score": round(score, 4),
            # Presence
            "frames_above_floor": p.frames_above_floor,
            "total_frames": p.frames,
            "presence_fraction": round(p.presence_fraction, 4),
            "present_term": round(present, 4),
            "presence_gate": round(presence_gate, 4),
            # Level, measured only where the sound is present
            "active_level_dbfs": round(p.active_level_dbfs, 2),
            "active_level_p90_dbfs": round(p.active_level_p90_dbfs, 2),
            "event_mean_rms_dbfs": round(p.mean_rms_dbfs, 2),
            # Spectral, also only over above-floor frames
            "active_flatness": round(p.active_flatness, 4),
            "active_centroid_hz": round(p.active_centroid_hz, 1),
            "peakiness_above_max": round(p.peakiness_above_max, 1),
            "high_to_low_ratio_db": round(p.high_to_low_ratio_db, 2),
            # Onsets
            "impulsive_onsets": p.impulsive_onsets,
            "impulse_density": round(p.impulse_density, 5),
            "onset_density_per_second": round(p.onset_density_per_second, 3),
            "whisper_impulse_penalty": round(penalty, 4),
            # Individual features
            "quiet_term": round(quiet, 4),
            "broadband_term": round(broadband, 4),
            "high_lean_term": round(high_lean, 4),
            "above_floor_term": round(above_floor, 4),
            "modulated_term": round(modulated, 4),
            "not_harmonic_term": round(not_harmonic, 4),
        }

    def _rule_speech(self, p: EventProfile, duration: float):
        """Voiced speech: moderate harmonicity, mid-band energy, modulation.

        Whisper scores higher on the high-frequency and broadband terms, so the
        two labels compete rather than overlapping.
        """
        # Some harmonic structure, but not as narrow as a hum.
        voiced = _ramp(p.max_peakiness, 3.0, 25.0) * (1.0 - _ramp(p.max_peakiness, 60.0, 150.0))
        # Energy in the middle band.
        mid = _ramp(p.mean_mid_band_snr_db, 5.0, 18.0)
        modulated = _ramp(p.mean_modulation, 0.15, 0.45)
        sustained = _ramp(duration, 0.15, 1.0)
        above_floor = _ramp(p.peak_snr_db, 6.0, 16.0)
        # Speech is not dominated by the very low bands.
        not_low = 1.0 - _ramp(p.mean_low_band_snr_db, 18.0, 30.0)

        score = (
            0.22 * voiced
            + 0.22 * mid
            + 0.22 * modulated
            + 0.12 * sustained
            + 0.12 * above_floor
            + 0.10 * not_low
        )
        if score < 0.20:
            return None
        evidence = []
        details = {
            "mean_mid_band_snr_db": p.mean_mid_band_snr_db,
            "mean_modulation": p.mean_modulation,
            "max_peakiness": p.max_peakiness,
        }
        if voiced > 0.4:
            evidence.append("harmonic_structure")
        if mid > 0.4:
            evidence.append("mid_band_energy")
        if modulated > 0.4:
            evidence.append("speech_like_modulation")
        if sustained > 0.4:
            evidence.append("sustained_duration")
        return LabelScore(
            label=LABEL_SPEECH, score=score, evidence=tuple(evidence),
            details=details,
        )

    def _rule_footstep(self, p: EventProfile, duration: float):
        """A step: a sharp low-frequency onset, often repeated.

        Note what is *not* used as evidence: the crest factor.  Measured, a
        40 ms frame of a decaying footstep thump has a crest of about 3, which
        is *lower* than the surrounding noise, so crest cannot support a footstep
        claim and is not used to make one.  The evidence is the flux onset, the
        low-frequency energy, and any repetition.
        """
        low = _ramp(p.max_low_band_snr_db, 8.0, 25.0)
        onset = _ramp(p.impulsive_onsets, 0.0, 1.0)
        repetition, regularity = _repetition_score(p.onset_intervals)
        # A single step is short.  A *sequence* of steps is not, and must not be
        # penalised for its length: three steps over two seconds is a normal
        # walk, and docs section 4 explicitly says multiple events form a
        # walking sequence.  So the duration term is dropped once a sequence is
        # established.  Duration is the active part, never the stored span,
        # which always carries the pre-roll and post-roll.
        if repetition > 0.4:
            short = 1.0
        else:
            short = 1.0 - _ramp(duration, 1.2, 3.0)
        # Every onset-related term is gated on `low`.  A footstep is
        # specifically a *low*-frequency transient: an onset with no low
        # frequency content is a click, which the knock rule owns, not a step.
        # Without this gate, the edges of a whispered phrase looked like
        # footsteps purely because they were regular, and a high-frequency
        # whisper scored as a sequence of steps.
        #
        # ...and on the event being *bursty*.  Footsteps arrive as discrete
        # impacts separated by silence, so a sustained sound that merely
        # contains a few impacts is not a sequence of steps.  This is the exact
        # mirror of the whisper rule's presence gate: the two labels are told
        # apart by the same measured quantity, read from opposite directions,
        # which is why a whispered phrase containing consonant transients is
        # not read as footsteps.
        bursty = 1.0 - _ramp(p.presence_fraction, 0.40, 0.75)
        transient_structure = (0.60 * onset + 0.40 * repetition) * low * bursty

        # Deliberately no "not tonal" term.  Measured: a 40 ms frame of a
        # decaying 90 Hz thump is tonal, scoring at the top of the peakiness
        # range, so penalising tonality would penalise the very thing this rule
        # is meant to detect.  A sustained pure tone is not a footstep, but the
        # interference rule already owns that case and it gates on onsets.
        score = (
            0.34 * low
            + 0.34 * transient_structure
            + 0.16 * short
        )
        if score < 0.20:
            return None
        evidence = ["sharp_onset"]
        details = {
            "max_low_band_snr_db": p.max_low_band_snr_db,
            "onsets": float(p.impulsive_onsets),
            "repetition_regularity": regularity,
        }
        if low > 0.4:
            evidence.append("low_frequency_energy")
        if repetition > 0.4:
            evidence.append("repeated_onsets_in_sequence")
        if short > 0.4:
            evidence.append("short_duration")
        return LabelScore(
            label=LABEL_FOOTSTEP, score=score, evidence=tuple(evidence),
            details=details,
        )

    def _rule_knock(self, p: EventProfile, duration: float):
        """A knock or tap: a genuinely sharp, broadband transient.

        Unlike a footstep, this one *is* evidenced by crest, because a knock is
        a sharp impact rather than a slow thump.
        """
        sharp = _ramp(p.max_crest, 5.0, 14.0)
        broadband = _ramp(p.mean_flatness, 0.05, 0.3)
        short = 1.0 - _ramp(duration, 0.8, 2.0)
        score = 0.45 * sharp + 0.25 * broadband + 0.20 * short + 0.10 * _ramp(
            p.impulsive_onsets, 0.0, 1.0
        )
        if score < 0.25:
            return None
        evidence = ["sharp_onset"]
        details = {"max_crest": p.max_crest, "mean_flatness": p.mean_flatness}
        if sharp > 0.5:
            evidence.append("high_crest_transient")
        if broadband > 0.4:
            evidence.append("broadband_impact")
        return LabelScore(
            label=LABEL_KNOCK, score=score, evidence=tuple(evidence),
            details=details,
        )

    def _rule_scraping(self, p: EventProfile, duration: float):
        """Scraping: sustained, bright, and unsteady."""
        bright = _ramp(p.high_to_low_ratio_db, -4.0, 8.0)
        sustained = _ramp(duration, 0.4, 1.5)
        unsteady = _ramp(p.max_flux, 0.5, 1.2)
        # As with clothing, brightness and duration are both true of silence, so
        # energy is required before the label can be offered.
        present = _ramp(p.peak_snr_db, 5.0, 12.0)
        gate = (0.15 + 0.85 * present) * self._family_onset_gate(p)
        score = (
            0.32 * bright + 0.26 * sustained + 0.24 * unsteady + 0.18 * present
        ) * gate
        if score < 0.32:
            return None
        evidence = ["sustained_bright_unsteady"]
        return LabelScore(
            label=LABEL_SCRAPING, score=score,
            evidence=("sustained_sound", "high_frequency_weighted"),
            details={"high_to_low_ratio_db": p.high_to_low_ratio_db,
                     "max_flux": p.max_flux},
        )

    def _rule_clothing(self, p: EventProfile, duration: float):
        """Clothing: sustained, low-weighted, low peakiness, unimpulsive."""
        low = _ramp(p.mean_low_band_snr_db, 5.0, 20.0)
        sustained = _ramp(duration, 0.3, 1.2)
        soft = 1.0 - _ramp(p.max_crest, 3.0, 6.0)
        dull = 1.0 - _ramp(p.mean_centroid_hz, 700.0, 2000.0)
        # "Soft" and "dull" are both trivially true of silence, so without an
        # energy term this rule fired on an empty event and a blank profile came
        # out as clothing rather than unknown.
        present = _ramp(p.peak_snr_db, 5.0, 12.0)
        # Gated on presence.  "Soft", "dull" and "sustained" are all trivially
        # true of an empty event, so without a gate a blank profile came out as
        # clothing rather than unknown (docs section 7: never force a label).
        gate = (0.15 + 0.85 * present) * self._family_onset_gate(p)
        score = (
            0.22 * low + 0.18 * sustained + 0.14 * soft + 0.30 * dull
            + 0.16 * present
        ) * gate
        if score < 0.32:
            return None
        return LabelScore(
            label=LABEL_CLOTHING, score=score,
            evidence=("sustained_low_frequency_sound", "unimpulsive"),
            details={"mean_low_band_snr_db": p.mean_low_band_snr_db,
                     "max_crest": p.max_crest},
        )

    def _rule_movement(self, p: EventProfile, duration: float):
        """Movement: the diffuse fallback for people moving.

        docs section 5 allows a family of labels and explicitly permits an
        uncertain result, so this rule is deliberately the most permissive of
        the sustained ones and yields to the more specific labels above it.
        """
        any_energy = _ramp(p.peak_snr_db, 6.0, 15.0)
        sustained = _ramp(duration, 0.2, 0.9)
        not_tonal = 1.0 - _ramp(p.max_peakiness, 30.0, 120.0)
        # Gated on energy and on the absence of sharp onsets, for the same
        # reasons as the other family rules.
        gate = (0.15 + 0.85 * any_energy) * self._family_onset_gate(p)
        score = (0.40 * any_energy + 0.32 * sustained + 0.28 * not_tonal) * gate
        if score < 0.25:
            return None
        return LabelScore(
            label=LABEL_MOVEMENT, score=score * 0.85,
            evidence=("diffuse_broadband_activity",),
            details={"peak_snr_db": p.peak_snr_db},
        )


def _narrowness(p: EventProfile) -> float:
    """How narrow the spectrum is, in 0..1.

    Bandwidth relative to the centroid: a hum or a motor concentrates its energy
    in a small fraction of the spectrum, so the ratio is small.  Speech and
    noise spread energy widely, so the ratio approaches or exceeds 1.  The
    centroid can be near zero for a very quiet frame, so the guard returns 0
    (nothing to judge) rather than dividing by zero.
    """
    if p.mean_centroid_hz <= 1.0:
        return 0.0
    ratio = p.mean_bandwidth_hz / p.mean_centroid_hz
    # A small ratio is the narrow one, so the ramp runs downward.  A single
    # tone has a ratio near 0.1; energy spread evenly across the spectrum has
    # a ratio of 1 or more.
    return float(1.0 - max(0.0, min(1.0, (ratio - 0.10) / 0.90)))
