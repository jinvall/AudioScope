# Detection and Classification (Phase 3)

Turns continuous measurements into persistent, classified events.

The design intent is in docs/DETECTION_AND_CLASSIFICATION.md. This document
records what was built, the measurements behind its thresholds, and its known
limits. Read the specification for *what* it should do; read this for *how* it
behaves in practice, including where it falls short.

    app/detection/
        candidate.py   per-frame: did something happen?
        tracker.py     hysteresis state machine -> event spans
        classifier.py  what might it be? (allowed to say "unknown")
        worker.py      the detection thread
    app/events/
        event.py       the stored event and its metadata
        store.py       per-event directory, lossless WAV + metadata.json
        cli.py         python -m app.events

Detection and classification run on by default during capture:

    python -m app.capture --seconds 30 --events events
    python -m app.events --verbose
    python -m app.capture --no-detection        # skip it

---

## 1. The central idea: presence

The single most useful number the classifier has is `presence_fraction`:

    presence_fraction = frames_above_floor / total_frames

A frame counts as above floor when its SNR *relative to the adaptive noise
floor* is at least `minimum_event_snr_db` (6 dB). It is never an absolute
dBFS comparison, because 6 dB above a −20 dBFS room is an event and 6 dB above a
−60 dBFS room is nothing (AGENTS.md section 8).

This exists because event *averages* are misleading. Three footsteps in a quiet
room have a low mean RMS and a high mean spectral flatness — for exactly the
same reason a sustained whisper does, because the gaps between the steps pull
the average down. Whichever of those two a naive classifier sees first, it gets
wrong. Presence separates them by an order of magnitude:

| event | presence |
|---|---|
| empty room | 0.00 |
| single footstep | 0.04 – 0.09 |
| footstep sequence | 0.04 |
| sustained whispered phrase | 0.50 – 0.65 |

It is the mirror of what the *footstep* rule uses: the footstep rule is gated
down by sustained presence, so the two labels are told apart by the same
measured quantity read from opposite directions. A whispered phrase containing
consonant transients is therefore still a whisper, not a sequence of steps.

### Related measurements, all from above-floor frames only

`active_level_dbfs` (median RMS of present frames), `active_flatness`,
`active_centroid_hz`. The event-wide means are still recorded but are not used
for whisper evidence, because a silent gap contributes to a mean and must not
contribute to a statement about what the sound is *when it is present*.

Per-item 1–3 of the fix that introduced these is required reading for anyone
modifying the classifier: the point is not "low RMS" but "quiet **and**
sustained **and** speech-like".

---

## 2. Candidate detection

`candidate.py` answers only "did something happen?", never "what was it". That
separation is deliberate: docs/DETECTION_AND_CLASSIFICATION.md section 2 asks
for deliberately broad detection, because a missed quiet event cannot be
recovered but a spurious one can be discarded.

Indicators, combined as the **strongest single** value rather than a product —
a product would let one weak indicator veto the rest, which is the opposite of
the breadth wanted here:

| indicator | evidence | scale |
|---|---|---|
| `level` | overall SNR above the floor | absolute SNR, 6→18 dB |
| `low_band` | 80–250 Hz above *its own* floor | 6→15 dB |
| `high_band` | 2–4 kHz above its own floor | 6→12 dB |
| `modulation` | envelope modulation depth | 0.15→0.45 |
| `flux` | spectral flux ÷ its running median | 1.3×→2.5× |
| `impulsive` | crest factor | 5→15 |

### Two indicators needed baselines, not thresholds

Both were wrong on first implementation and the error was only visible in
measurement.

**Crest factor.** The indicator was set at 1.5, which flags *everything*.
Measured: Gaussian noise over a 1920-sample frame has a median crest of 3.6 and
a maximum of 4.6, so any threshold below 5 is not evidence of a transient. Worse,
a footstep thump measures **2.5–3.9** — *below* the noise around it — because a
40 ms frame holds only ~3.6 cycles of a decaying 90 Hz thump. So crest detects
sharp broadband impacts (clicks, taps, glass) and says nothing about footsteps.
The footstep classifier is therefore forbidden from using it as evidence, and
that is documented at the rule itself rather than left to be rediscovered.

**Spectral flux.** An absolute threshold is useless: uncorrelated background
noise already produces a flux of ~0.376, because successive noise spectra are
uncorrelated and half the rectified difference is large. Measured, a footstep
onset spikes to 0.999 — a factor of 2.7. Flux is only meaningful against its own
running median, so that is what the indicator uses.

**Modulation depth** is `(max−min)/max` over a 1 s window. The textbook index
divides by the mean, which is unbounded: measured, it ran past 12 for a footstep
sequence and saturated, so every active sound scored identically. Dividing by the
peak gives 0.07 for quiet noise and ~0.98 for either footsteps or speech — bounded
and honest, and it deliberately does *not* separate those two, because presence
does.

---

## 3. Event tracking

`tracker.py` implements the hysteresis of docs/AUDIO_PIPELINE.md section 8, so
one real event does not produce a candidate on every other frame:

| parameter | default | role |
|---|---|---|
| `onset_threshold` | 0.30 | higher bar to start |
| `continuation_threshold` | 0.16 | lower bar to keep going |
| `release_timeout` | 0.40 s | quiet time before the event ends |
| `min_duration` | 0.05 s | rejects blips |
| `max_duration` | 30 s | splits a runaway event |
| `merge_gap` | 0.30 s | folds two detections that are one thing |

`continuation_threshold` must be lower than `onset_threshold`, enforced by
config validation: equal thresholds would make the state machine chatter, which
is the exact failure the section exists to prevent.

### One frame of latency, on purpose

A completed event is held for one frame before being emitted, because merging
cannot be decided until the *next* detection is known. That is the only latency
the tracker adds.

### Pre-roll and post-roll

The stored span is `onset − pre_roll` to `last_active + post_roll` (5 s each by
default). Two consequences are easy to get wrong and both are handled:

- **Pre-roll** comes from the ring buffer. If the buffer did not hold it, the
  shortfall is recorded rather than the event silently starting later.
- **Post-roll is in the future** for an event detected while the stream is
  running. Extracting immediately produced an event whose stored audio was
  several seconds shorter than its recorded span while still reporting the
  pre-roll as complete — an event claiming 11.70 s and containing 6.68 s. The
  detection worker now holds a completed event until capture has passed its end
  sample, and reports `postroll_missing_frames` when the stream ended first.
  `metadata.json` carries `span_duration_seconds` and
  `stored_duration_seconds` separately so the two can be compared rather than
  assumed equal.

---

## 3a. Event segmentation

Segmentation answers: *where does one event end and the next begin?* It is kept
strictly generic - no sound class is named anywhere in the tracker, and a test
enforces that (see section 7).

### Why it was needed

Termination originally depended on one thing: the activation dropping below
`continuation_threshold` for `release_timeout`. On a continuously active signal
that never happens, because activation is dominated by level *relative to the
adaptive floor*, and a continuous signal never falls back to it. Worse, **every
detected onset reset the release timer**, so with onsets arriving more often
than the release period, the timer was structurally unable to reach its
threshold.

A second defect compounded it: `max_duration` did split long events, but the
very next frame began a new event whose onset was ~0 s after the previous one's
last active frame, so `merge_gap` re-fused them immediately. Measured on a
synthetic 60 s continuous signal, the 30 s cap fired twice and the pieces were
merged straight back, yielding a single 60 s event. The cap had no net effect.

### Boundaries

Two conditions, both independent of silence:

| condition | measures | ends the event when |
|---|---|---|
| `onset_cluster` | inter-onset structure | onsets further apart than `onset_cluster_window`, once at least `min_onsets_for_clustering` have been seen |
| `acoustic_change` | spectral character | the recent window's centroid or flatness departs from the prior window by more than `change_point_sensitivity` spreads **or** `change_point_relative` of the prior level |

`min_onsets_for_clustering` is a guard that *enables* the check, not a rule that
defines an event: a signal with one or two onsets is never fragmented by it.

Every value is explicit configuration in `DetectionConfig`: `onset_cluster_window`,
`min_onsets_for_clustering`, `change_point_window`, `change_point_sensitivity`,
`change_point_relative`, `max_merge_interval`. None is hard-coded.

### Two details that were wrong first time

**A single trailing window cannot detect its own transition.** Comparing the
current value against a window that immediately absorbs it makes the boundary
unreachable, because the spread grows with the very change being measured. The
detector therefore uses two windows: the older part of the trail is the prior
state, the newest third is the current state, and the spread comes from the
prior part only. On a clean step this fires within one window.

**The change window must be full before a change can be claimed.** The first
frames of any event are exactly where the spectrum legitimately differs from the
background - that is the onset, not a change of character. With a partially
filled window the detector fired at the start of real sounds and produced 80 ms
fragment events. Detecting a change requires history, so the window must be
filled first.

### Merging can no longer undo a boundary

`max_merge_interval` bounds merging, and an event whose termination was a
deliberate boundary (`acoustic_change`, `onset_cluster`, `max_duration`) is
never merged into its neighbour. The prohibition lives on the *event*, not on
the tracker: a tracker-wide flag is cleared when the next event begins, so the
boundary that produced the event was forgotten before the merge decision was
made, and the pieces were fused again.

### Termination reasons

`EventTermination` records *why* each event ended, and is stored in
`metadata.json`:

    quiet | acoustic_change | onset_cluster | max_duration | end_of_stream

`quiet` is the interesting one operationally: it means activity genuinely
stopped. The other three can fire while the signal is fully active, which is
the point.

`EventTracker.transitions` logs state changes only - start, terminate, merge,
merge_refused - never one record per frame, and is bounded. Each carries the
measurement that justified it, so a boundary can be audited rather than
guessed at:

    {'action': 'terminate', 'at_seconds': 15.04,
     'reason': 'acoustic_change', 'feature': 'centroid_hz',
     'recent_mean': 1120.0, 'prior_mean': 800.0, 'prior_spread': 0.0,
     'deviation': 320.0, 'threshold': 280.0}

---

## 4. Classification

`classifier.py`. Rules for whisper, speech, footstep, knock, movement,
clothing, scraping, interference, clipping, and unknown.

### Confidence is always `null`

Not an omission. docs/DETECTION_AND_CLASSIFICATION.md section 8 requires
confidence to come from an actual calibrated model and forbids "arbitrary
cosmetic scaling". What is here *is* a classifier and its score is a real,
reproducible number, so it is reported as `label_scores` with the evidence and
the margin — but a rule-based score is not a calibrated probability, and
presenting it as a confidence percentage is exactly the cosmetic number the
specification prohibits. Every `metadata.json` states this in
`confidence_note`.

### A note on `possible_*`

Labels are prefixed `possible_` because that is what a rule-based classifier
can honestly claim. Nothing here identifies a whisper; it says the evidence is
consistent with one, lists what that evidence was, and names the runner-up.

### Thresholds that came from measurement

| rule | threshold | provenance |
|---|---|---|
| interference | mains match within 6% | at 12% a 90 Hz footstep matched 100 Hz mains; a real 50 Hz hum lands on bin 2.0 exactly, so tightness costs nothing |
| interference | peakiness ≥ 60 | measured: noise 13, harmonic stack 213, pure tone 541–640 |
| interference | onsets ≥ 2 ⇒ reject | hum is steady; repeated sharp onsets are impacts whatever the pitch |
| footstep | short-duration term dropped for sequences | three steps over two seconds is a normal walk (section 4) |
| tonal | peakiness ≥ 100 | same measurement as interference |
| whisper | presence 0.15→0.45 | measured separation: footsteps 0.04, whisper 0.50–0.65 |

### Whisper rule

    quiet + sustained presence + speech-like temporal structure

with a graded impulse-density penalty. There is **no** `impulsive_onsets > 0`
rejection: whispered speech has consonant transients ("t", "k", "p", "sh", "ch"
all produce short impulses), and a hard gate would make a real whisper
unclassifiable. `impulse_density` is `onsets / max(frames, 1)` — per *frame*, not
per second, so it does not change meaning if the analysis hop is reconfigured —
and it subtracts a bounded amount (max 0.30) rather than returning nothing.

A test inspects the rule's source to stop that hard gate being reintroduced.

### Diagnostics

`label_details` in `metadata.json` and `--verbose` expose, per label, the
measured terms behind the score. For the whisper rule that is:
`presence_fraction`, `frames_above_floor`, `total_frames`, `active_level_dbfs`,
`active_flatness`, `impulsive_onsets`, `impulse_density`, `whisper_score`, plus
each individual term and the presence gate. The point is that a future false
positive can be *explained from measurements* rather than debugged by guessing
thresholds.

---

## 5. A change to the noise floor that Phase 3 forced

Phase 3 could not work on the Phase 2 floor, and the reason is worth recording.

Phase 2 estimated the floor as the **mean** of accepted sub-window levels,
chosen over a minimum to avoid a narrow-band bias. That works for intermittent
events but has a fatal consequence for sustained ones: the mean of a window is
by construction the average level of whatever occupied it, so **any sound
present most of the time defines the floor and its reported SNR collapses to
zero**. Measured, a whispered phrase occupying 71% of a window scored presence
0.00 at *every* amplitude tried, from 7 dB to 24 dB above room noise.

The estimate is now a **low quantile (0.20)** of the sub-window distribution,
taken in linear power, with a 0.1 s sub-window. Measured:

| sub-window | whisper presence | footsteps | empty room | phantom band SNR |
|---|---|---|---|---|
| 0.50 s (was) | 0.11 | 0.04 | 0.00 | 0.02 dB |
| 0.10 s (now) | 0.65 | 0.04 | 0.00 | 0.19 dB |

A quantile still finds the quiet part of the window, so a sustained sound stays
visible, while averaging over the ring of sub-windows first keeps it far less
biased across band widths than a minimum. The last two columns are the check
that this did not loosen anything else: footsteps and an empty room are
unchanged.

**Cost:** analysis went from 0.09 to ~0.63 of one core. The quantile is
computed once per sub-window per band and cached, which is what keeps it
affordable.

---

## 6. What a stored event looks like

    events/2026-09-27/event_000000/
        metadata.json
        original.wav

`original.wav` is the unmodified captured audio — no filtering, no
normalisation — written atomically, so a partially written event never looks
complete. `metadata.json` is written *after* the audio, so it always refers to
audio that exists.

Read them back with:

    python -m app.events
    python -m app.events --verbose
    python -m app.events --json
    python -m app.events --event event_000003

---

## 7. Known limits

Stated plainly, because a monitoring tool that hides its own weaknesses is
worse than useless.

**Segment boundaries are only as good as the change features.** Segmentation
now works on inter-onset structure and spectral change, but only two features
are watched: spectral centroid and flatness. A signal that changes character
without moving either - steady level, steady timbre, but a different
*rhythm* - will not be segmented. Adding features to watch is the obvious
extension.

**Segmentation changed the meaning of `presence`, and this interacts with
classification.** With events properly bounded, a short burst now has
`presence` near 1.0, because within its own short event every frame is above
the floor. The whisper rule uses presence to mean *sustained*, so it has to be
read as "sustained relative to the event's own span". A footstep *sequence*
still scores low (0.10) because its gaps live inside one event, which is the
case the rule was written for; an isolated single burst now looks sustained.
That is a real limitation of the interaction, not a solved problem.

**Boundary detection needs ~1.5 s of event before it can act.** That is the
`change_point_window` and it is deliberate, but it means events shorter than the
window can only be ended by `quiet`, onset clustering, or the duration cap.

**Events longer than the ring buffer lose their pre-roll.** The buffer holds 30 s
by default, so a longer event reports a large `preroll_missing_frames` and its
stored audio is clipped to what the buffer still held. This is reported, not
hidden, but it means long events cannot have their full pre-roll recovered.

**Input overflow on this host's audio device.** The ALC897-through-PulseAudio path
drops samples at roughly 5 per 10 s *with no processing running at all*, rising to
15–20 with analysis. Block size is not the cause (measured 256→42, 512→20,
1024→30, 4096→62 overflows over three 8–10 s runs — no trend). The loss is
counted and reported as a warning, and the recording is labelled non-contiguous,
but the audio really is missing. This is an environment characteristic, not
something the application can tune away.

**A signal present 100% of the time is invisible.** No adaptive floor can see it,
because there is no silent frame to anchor to. Real whispers have pauses; a
perfectly continuous hum is treated as the room.

**The classifier has never heard a real whisper.** All whisper validation is
synthetic. The synthetic signal needed room noise *and* phrase gaps to be
detectable at all, and it was calibrated against measured behaviour rather than
tuning the classifier to pass an easy case. That is not a substitute for
recording someone actually whispering, and until that is done the whisper label
should be read as "consistent with" rather than "is".

**Headroom is thin.** Analysis ~0.63 + detection ~0.11 ≈ 0.74 of one core, against
a 1.0 budget. That leaves little for the GUI and nothing for separation. The
honest options are a lower analysis rate or more cores, and picking either is a
decision about what to give up.
