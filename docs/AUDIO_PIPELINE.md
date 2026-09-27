# Audio Microscope — Audio Pipeline

## 1. Input

Preferred internal format:

    48,000 Hz
    mono
    float32

If the hardware differs, convert once at the input boundary.

---

# 2. Capture

The capture callback must:

1. receive audio
2. convert format if necessary
3. append to ring buffer
4. enqueue lightweight analysis data

Nothing expensive belongs here.

---

# 3. Ring Buffer

The ring buffer provides historical context.

Default:

    30 seconds

Configurable:

    10–120 seconds

The buffer must support:

- append
- snapshot
- time-range extraction
- pre-roll extraction
- thread-safe reads

---

# 4. Analysis Frames

The analysis worker operates on short overlapping frames.

Frame duration should be configurable.

Typical initial value:

    20–50 ms

with overlap sufficient for stable spectral analysis.

Do not make detector timing dependent on a specific arbitrary frame size.

---

# 5. Features

Calculate:

- RMS
- peak
- crest factor
- zero crossing rate
- spectral centroid
- bandwidth
- rolloff
- spectral flatness
- spectral flux
- band energy

Frequency bands:

    20–80 Hz
    80–250 Hz
    250–500 Hz
    500–1000 Hz
    1000–2000 Hz
    2000–4000 Hz
    4000–8000 Hz
    8000–16000 Hz
    16000–24000 Hz

---

# 6. Adaptive Noise Floor

Maintain a slowly changing estimate.

The estimator should avoid learning from strong transient events.

Recommended conceptual behavior:

    quiet environment
        ↓
    noise floor follows slowly

    sudden sound
        ↓
    detector responds immediately
        ↓
    noise floor does NOT immediately jump upward

Use different attack and release behavior.

---

# 7. Candidate Detection

A candidate may be created from:

- sudden energy increase
- spectral flux
- unusual frequency-band change
- broadband transient
- modulation pattern
- persistent low-level energy
- harmonic change
- unusual spectral shape

Multiple indicators should be combinable.

---

# 8. Event Hysteresis

Avoid:

    candidate
    no candidate
    candidate
    no candidate

for every frame of one real event.

Use:

- onset threshold
- continuation threshold
- minimum duration
- maximum duration
- release timeout
- merge gap

---

# 9. Pre-Roll

When an event begins at time T:

    event_start = T - pre_roll

provided the ring buffer contains enough history.

---

# 10. Post-Roll

Do not immediately finalize an event when its first candidate frame ends.

Wait for the configured release period.

Then append post-roll.

This captures trailing sounds.

---

# 11. Spectrogram

The spectrogram should use the same event time base as the waveform.

The UI should be able to display:

- frequency
- time
- intensity

Event markers must align with the waveform and spectrogram.

---

# 11a. Event Fingerprint and Index

After an event is closed by the tracker (section 8) and extracted with its
pre-roll and post-roll (sections 9 and 10), the event is fingerprinted and
indexed.

The fingerprint is built entirely from measurements already made upstream:

    timing     from the tracker's onset and last-active samples
    energy     from the per-frame level and SNR accumulated per event
    spectral   from the per-frame centroid and flatness accumulated per event
    temporal   from the onset counters and intervals the tracker already keeps
    bands      from the per-band SNR summaries

No transform is added and the audio is not re-read. The per-frame accumulators
are four arithmetic folds and one bounded append, and they cost nothing
measurable on the detection thread.

The result is a fixed-size summary - 23 normalised comparison fields and a
16-point energy envelope, about 1.5 kB - carrying three versions so that a
record remains interpretable after the detector evolves.

Persistence runs on its own thread. The detection thread hands over the event
and continues; a slow disk can therefore delay indexing but can never delay
capture or event extraction. If the handover queue fills, the event is dropped
*from the index* and counted: its audio and full metadata are already on disk,
so only the index entry is late, and that is reported rather than hidden.

Human decisions are stored separately from all of this and never overwrite it.
See `docs/EVENT_DATABASE.md`.

---

# 12. Playback

Playback must use actual captured/generated audio.

Never synthesize a fake waveform for demonstration.

---

# 13. Test Input

A WAV file must be able to enter the same analysis path.

Example:

    python -m app.test test.wav

The test mode should avoid duplicating detector logic.

It should feed the same pipeline used by live input.
