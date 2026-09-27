# AGENTS.md — Audio Microscope Development Contract

## 1. Mission

Build a Linux desktop application that continuously listens to a live microphone/audio input and can:

1. Detect acoustic events.
2. Identify likely event types.
3. Capture the event with surrounding context.
4. Isolate a requested sound from the mixed recording.
5. Let the user hear the isolated sound.
6. Let the user save the isolated sound.
7. Preserve the original recording as evidence.

The application is intended to detect and isolate sounds such as:

- whispers
- quiet speech
- speech
- footsteps
- people moving
- clothing/rubbing sounds
- knocks
- taps
- impacts
- scraping
- object movement
- mechanical sounds
- electrical interference
- hum
- buzz
- clipping
- dropouts
- unknown/unclassified sounds

The central requirement is:

> DETECTION IS NOT ENOUGH. THE USER MUST BE ABLE TO HEAR THE ISOLATED SOUND.

A detector that correctly says "footstep detected" but cannot produce an isolated footstep that the user can listen to is incomplete.

---

## 2. Non-Negotiable Requirements

### 2.1 Preserve the original audio

Never destroy, overwrite, or permanently modify the captured source audio.

Processing must operate on copies/buffers.

The original event recording must remain available.

---

### 2.2 CPU-only operation

The development machine does not have a GPU.

Do not assume:

- CUDA
- NVIDIA
- GPU acceleration
- TensorRT
- GPU-only PyTorch operations

The application must function on CPU.

If an optional GPU acceleration path exists, CPU remains the required baseline.

---

### 2.3 Never block live audio capture

The real-time audio callback must remain extremely lightweight.

Do NOT perform inside the audio callback:

- PyTorch inference
- source separation
- disk I/O
- WAV encoding
- large allocations
- GUI operations
- network operations
- expensive FFT processing
- model loading

Use queues, ring buffers, and worker threads/processes.

The live capture pipeline has priority over analysis and separation.

---

### 2.4 No fake features

Do not implement fake or simulated versions of required functionality.

Forbidden examples:

- A "separate" button that merely applies a bandpass filter.
- A "whisper detector" that is actually only a VAD.
- A confidence number invented from arbitrary thresholds.
- A fake AudioSep interface that does not call a real model.
- A UI waveform that does not represent actual audio.
- A playback button that does not play the generated file.
- A save button that does not actually save the result.
- Placeholder separation output containing the original mix.

If a feature cannot currently be implemented, expose the limitation clearly rather than pretending it works.

---

## 3. Source Separation Is a Core Feature

The application must support query-based source separation.

Examples:

    "a whisper"
    "footsteps"
    "a person moving"
    "electrical interference"
    "a knock"
    "scraping"
    "a mechanical sound"

The user must be able to:

1. Select an event.
2. Enter or select a separation query.
3. Run source separation.
4. Receive a real audio output.
5. Play the output.
6. Save the output.

The source separation implementation must use an actual supported model/API.

Do not substitute ordinary filtering and call it source separation.

---

## 4. Automatic Detection vs Manual Isolation

Both are required.

### Automatic path

    microphone
      ↓
    rolling buffer
      ↓
    real-time analysis
      ↓
    candidate event
      ↓
    event classification
      ↓
    event window
      ↓
    source separation
      ↓
    isolated audio

### Manual path

The user must be able to select a region of the waveform and request separation manually.

Manual isolation is required even when automatic detection fails.

---

## 5. Event Context

Default event capture:

- 5 seconds pre-event
- event duration
- 5 seconds post-event

These values must be configurable.

The rolling buffer should default to approximately 30 seconds.

Allowed configuration range:

- 10–120 seconds

The system must ensure enough history exists to extract the configured pre-roll.

---

## 6. Audio Format

Prefer:

- mono
- float32
- 48 kHz

If the input device uses another format, resample once into the internal format.

Avoid repeated resampling.

---

## 7. Real-Time Analysis

Use inexpensive analysis for continuous monitoring.

Possible features include:

- RMS
- peak
- crest factor
- zero-crossing rate
- spectral centroid
- spectral bandwidth
- spectral rolloff
- spectral flatness
- spectral flux
- band energies
- adaptive noise floor

Useful frequency bands:

- 20–80 Hz
- 80–250 Hz
- 250–500 Hz
- 500–1 kHz
- 1–2 kHz
- 2–4 kHz
- 4–8 kHz
- 8–16 kHz
- 16–24 kHz

The detector should react to changes relative to the environment rather than relying exclusively on fixed dBFS thresholds.

---

## 8. Adaptive Noise Floor

The system must learn the current acoustic environment.

Do not assume silence is:

    -60 dBFS

or any other fixed value.

Maintain an adaptive noise estimate.

The noise model should have asymmetric attack/release behavior so that a sudden event does not immediately become part of the noise floor.

---

## 9. Whisper Detection

Whispers are especially important.

A whisper may have:

- low RMS
- little/no fundamental frequency
- increased high-frequency energy
- broadband energy
- speech-like temporal modulation
- weak harmonic structure
- low overall signal-to-noise ratio

Do not use volume alone.

A quiet signal must not automatically be classified as silence.

Possible labels:

- whisper
- quiet speech
- speech
- possible speech

These labels should represent evidence from the actual signal.

---

## 10. Footstep Detection

Footsteps should consider:

- transient onset
- low-frequency energy
- broadband energy
- spectral shape
- duration
- attack/release
- temporal spacing
- repeated events

A sequence of similar transients separated by plausible intervals may provide stronger evidence than one isolated transient.

Do not require a fixed walking cadence.

---

## 11. Movement Detection

Movement may produce:

- clothing noise
- rubbing
- scraping
- surface contact
- object handling
- shifting weight
- chair/furniture movement
- low-frequency body motion
- irregular broadband changes

Movement detection should allow uncertain/unknown classification.

Do not force every event into a known class.

---

## 12. Interference Detection

Analyze for:

- DC offset
- 50/60 Hz hum
- harmonics
- electrical buzz
- periodic interference
- clipping
- dropouts
- discontinuities
- buffer artifacts
- broadband bursts
- repetitive electronic noise

Interference should be distinguishable from acoustic events where possible.

---

## 13. Unknown Events

Unknown is a legitimate result.

Never force an unfamiliar sound into an existing category merely to produce a label.

The UI should be able to display:

    Unknown acoustic event

with supporting measurements.

---

## 14. Separation Processing

Separation must run asynchronously.

Default:

    1 separation worker

The system should prevent source separation from starving:

- live capture
- waveform updates
- event detection
- user interaction

If the CPU is overloaded, separation may queue.

Live monitoring must continue.

---

## 15. Multiple Separation Attempts

An event may be separated more than once.

Example:

    Event 000123

    separation_001/
        isolated.wav
        metadata.json

    separation_002/
        isolated.wav
        metadata.json

The second attempt must not overwrite the first.

This allows the user to try:

    "a whisper"

then:

    "a person speaking"

then:

    "background mechanical noise"

on the same event.

---

## 16. Audio Outputs

Every event should support:

### Original

The untouched event recording.

### Isolated

The raw output from the source separation system.

### Enhanced

An optional processed version of the isolated signal.

Enhancement must never replace the original isolated result.

---

## 17. Enhancement Rules

Optional enhancement may include:

- DC removal
- conservative normalization
- clipping prevention
- gentle cleanup

Do not aggressively denoise automatically.

Aggressive processing can destroy the acoustic evidence being investigated.

---

## 18. GUI Requirements

The GUI must provide:

- live waveform
- spectrogram
- event timeline
- event markers
- event list
- event detail panel
- playback controls
- separation controls
- query input
- save controls
- CPU/RAM monitoring
- processing queue status

Playback controls must include:

- Play
- Pause
- Stop
- Seek
- Loop
- Volume

Allow A/B comparison between:

- Original
- Isolated
- Enhanced

---

## 19. Manual Region Selection

The waveform must support selecting a time range.

The user can then:

1. Select the range.
2. Enter a separation query.
3. Run separation.
4. Listen to the result.
5. Save it.

This path must not depend on automatic event detection.

---

## 20. Evidence Storage

Use WAV for lossless evidence.

Each event should have its own directory.

Example:

    events/
      2026-09-25/
        event_000001/
          metadata.json
          original.wav
          separation_001/
            isolated.wav
            metadata.json

---

## 21. Metadata

At minimum record:

- event ID
- timestamp
- duration
- classification
- confidence, only if legitimately calculated
- noise floor
- peak level
- sample rate
- channels
- input device
- separation query
- separation model
- processing time
- output filename

Do not invent values.

---

## 22. Continuous Recording

Optional continuous recording should use chunked files.

Default chunk:

    15 minutes

The original continuous stream must not be altered by event processing.

---

## 23. Test Mode

The same analysis pipeline must work with a WAV file.

Example:

    python -m app.test input.wav

This allows repeatable development without requiring live microphones.

---

## 24. CLI Requirements

Provide:

    ./run.sh

    python -m app.test input.wav

    python -m app.separate input.wav \
        --query "a whisper" \
        --output isolated.wav

---

## 25. Development Order

Implement in this order:

1. Audio capture
2. Ring buffer
3. Playback
4. Basic DSP
5. Adaptive noise floor
6. Candidate event detection
7. Event tracking
8. Classification
9. Source separation
10. Event extraction
11. GUI
12. Recording/storage
13. Performance monitoring
14. Integration tests

Do not build a large GUI around nonfunctional backend features.

---

## 26. Required Proof

The project is not complete until this sequence works:

    live microphone
       ↓
    quiet sound occurs
       ↓
    event detected
       ↓
    event captured
       ↓
    user selects event
       ↓
    separation query entered
       ↓
    source separation runs
       ↓
    isolated WAV created
       ↓
    isolated WAV plays
       ↓
    isolated WAV can be saved

Repeat this for:

- whisper
- footsteps
- movement
- interference
- unknown sound
- manually selected region

---

## 27. Agent Behavior

The coding agent must:

- inspect existing code before changing it
- make small verifiable changes
- run tests after meaningful changes
- report actual failures
- never claim success without testing
- preserve working functionality
- avoid unnecessary rewrites
- avoid speculative abstractions
- document real limitations

When choosing between a complicated theoretical architecture and a simpler working implementation, prefer the simpler implementation that satisfies the actual requirements.

---

## 28. Definition of Done

A feature is done only when:

1. It exists in the backend.
2. It is connected to the actual pipeline.
3. It has tests where practical.
4. The GUI action actually performs the operation.
5. The result is observable.
6. Errors are handled.
7. The feature works on CPU.
8. Original audio remains preserved.

No placeholders should remain in the production path.
