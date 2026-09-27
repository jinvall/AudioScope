# Detection and Classification

## 1. Philosophy

Detection answers:

    "Something happened."

Classification answers:

    "What kind of acoustic event might it be?"

Separation answers:

    "Can we extract the requested sound from the mixture?"

These are separate operations.

---

# 2. Candidate Detection

Candidate detection should be deliberately broad.

It is better to produce:

    Unknown event

than to miss a quiet event because the detector was too restrictive.

---

# 3. Whisper

Whispers can be difficult because they may have very low energy.

Useful characteristics:

- weak fundamental
- weak harmonic structure
- broadband energy
- high-frequency content
- speech-like modulation
- low RMS relative to normal speech

The classifier should compare these features with the adaptive noise floor.

Do not equate low volume with silence.

---

# 4. Footsteps

Consider:

- transient energy
- low-frequency energy
- broadband response
- duration
- onset
- decay
- repetition

Multiple events can form a walking sequence.

Do not require a fixed number of steps.

---

# 5. Movement

Movement may be diffuse rather than impulsive.

Look for:

- changing broadband energy
- rubbing
- scraping
- intermittent contact
- low-frequency movement
- repeated irregular patterns

Possible labels:

    person_moving
    clothing
    rubbing
    scraping
    object_movement
    unknown_movement

---

# 6. Interference

Look for:

- narrow spectral peaks
- harmonics
- periodicity
- 50/60 Hz energy
- clipping
- discontinuities
- repeated digital artifacts

Interference classification should consider whether the signal is acoustically plausible or electrically structured.

---

# 7. Unknown

Unknown is valid.

Do not force classification.

Store the observed features so the event can be investigated manually.

---

# 8. Confidence

Confidence must be based on actual model/classifier output.

Never generate confidence from:

    random()
    
or arbitrary cosmetic scaling.

If no calibrated confidence exists, display a qualitative evidence state instead.

---

# 9. Classification Output

Recommended structure:

    classification
    confidence
    evidence
    features
    timestamp

Example:

    classification:
        possible_whisper

    evidence:
        low_rms
        elevated_high_frequency_energy
        speech_like_modulation

This is evidence, not certainty.

---

# 10. Classification and Separation

Classification should suggest useful separation queries.

For example:

    whisper
        → "a whisper"

    footsteps
        → "footsteps"

    movement
        → "a person moving"

    interference
        → "electrical interference"

The user must be able to override the suggested query.
