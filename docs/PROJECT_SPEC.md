# Audio Microscope — Project Specification

## 1. Purpose

Audio Microscope is a Linux desktop application for continuous acoustic monitoring.

Its purpose is not merely to detect sounds.

Its purpose is to detect, investigate, isolate, hear, and preserve acoustic events occurring inside a mixed live audio stream.

The primary workflow is:

    LIVE AUDIO
        ↓
    CONTINUOUS BUFFER
        ↓
    REAL-TIME ANALYSIS
        ↓
    EVENT DETECTION
        ↓
    EVENT CLASSIFICATION
        ↓
    EVENT EXTRACTION
        ↓
    SOURCE SEPARATION
        ↓
    ISOLATED AUDIO
        ↓
    PLAY / SAVE

---

# 2. Primary Use Case

The microphone hears a mixture such as:

    room ambience
    HVAC
    computer fans
    traffic
    electrical noise
    distant speech
    footsteps
    movement
    mechanical sounds

The application should allow the user to determine:

- What happened?
- When did it happen?
- What evidence supports the classification?
- Can the sound be isolated?
- Can the isolated sound be heard?
- Can it be saved?

---

# 3. Core Event Types

The initial system should recognize or investigate:

### Human-related

- whisper
- quiet speech
- speech
- footsteps
- person moving
- clothing noise
- rubbing
- breathing-like sounds

### Physical interaction

- knock
- tap
- impact
- scrape
- object movement
- surface contact
- handling

### Mechanical

- motor
- fan
- vibration
- machine noise
- repetitive mechanical noise

### Electrical/interference

- 50 Hz hum
- 60 Hz hum
- harmonic hum
- electrical buzz
- periodic interference
- digital artifact
- clipping
- dropout

### Unknown

Anything that does not have enough evidence for a known category.

---

# 4. Automatic Detection

Automatic detection should continuously monitor the live stream.

The detector should operate at low computational cost.

It should identify candidate events rather than attempting expensive source separation continuously.

Candidate detection can use:

- energy changes
- spectral flux
- band changes
- transients
- modulation
- harmonic structure
- broadband changes
- persistent low-level changes
- unusual spectral patterns

---

# 5. Event Window

Default:

    pre-roll  = 5 seconds
    post-roll = 5 seconds

These are configurable.

The event system must be capable of capturing the event before the detector knows the event exists.

This is why the rolling buffer is required.

---

# 6. Isolation

Isolation is a first-class feature.

The application should use query-based source separation when appropriate.

Example:

    Input:
        room + HVAC + footsteps + speech

    Query:
        "footsteps"

    Output:
        footsteps.wav

The output must be real generated audio.

A filtered copy of the original signal is not considered source separation.

---

# 7. Manual Investigation

Automatic classification will never be perfect.

The user must be able to select any region manually.

Example:

    02:13:41.250 → 02:13:46.750

Then enter:

    "a whisper"

and run separation.

---

# 8. User Experience

The main screen should expose three simultaneous concepts:

### What is happening now?

Live waveform/spectrogram.

### What happened?

Event timeline/list.

### What can I hear?

Playback and isolated outputs.

The user should not have to navigate through unrelated screens to get from an event to its audio.

---

# 9. Audio Views

Each event can contain:

### Original

Untouched recording.

### Isolated

Direct source-separation output.

### Enhanced

Optional post-processed version.

Never discard the original.

---

# 10. Storage

WAV is the preferred evidence format.

Example:

    events/
      2026-09-25/
        event_000001/
          metadata.json
          original.wav

          separation_001/
            isolated.wav
            metadata.json

          separation_002/
            isolated.wav
            metadata.json

---

# 11. Performance

The application must prioritize:

1. Audio capture
2. Event detection
3. User interface responsiveness
4. Source separation

Source separation is expensive and may run asynchronously.

The application must continue capturing audio while separation occurs.

---

# 12. CPU-Only Requirement

The baseline environment is CPU-only.

The application must not require:

- NVIDIA GPU
- CUDA
- GPU-specific drivers

Model loading and inference must have a CPU-compatible path.

---

# 13. Failure Handling

If source separation fails:

- preserve the original event
- preserve metadata
- report the failure
- allow another attempt

A separation failure must never destroy the captured evidence.

---

# 14. Acceptance Criteria

The project passes acceptance when a user can:

1. Start live monitoring.
2. See live audio.
3. See a detected event.
4. Open the event.
5. Play the original.
6. Enter a separation query.
7. Generate an isolated result.
8. Play the isolated result.
9. Save the result.
10. Repeat the process for another query.

Manual region isolation must also work.

---

# 15. Explicit Non-Goals

The first version is not required to:

- identify every sound perfectly
- provide forensic certainty
- determine the identity of a person
- determine the source location of a sound
- continuously run expensive neural inference on every audio frame

The system should instead provide useful acoustic evidence and real source-isolated audio.
