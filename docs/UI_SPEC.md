# UI Specification

## 1. Main Window

The main window should expose:

- live waveform
- live spectrogram
- event timeline
- event list
- selected event details
- playback controls
- separation controls
- system status

---

# 2. Live Waveform

Display:

- current audio
- amplitude
- time scale
- event markers

The waveform must represent actual captured audio.

---

# 3. Spectrogram

Display:

- frequency
- time
- magnitude

The spectrogram should scroll during live operation.

---

# 4. Event Timeline

Events should appear chronologically.

Each event should show:

- timestamp
- duration
- classification
- confidence/evidence
- processing state

---

# 5. Event Detail

Selecting an event should show:

    Event ID
    Time
    Duration
    Classification
    Noise floor
    Peak
    Input device
    Sample rate

Also show:

    Original
    Isolated outputs
    Enhanced outputs

---

# 6. Playback

Controls:

    Play
    Pause
    Stop
    Seek
    Loop
    Volume

The user should be able to switch between:

    Original
    Isolated
    Enhanced

---

# 7. Separation Panel

Provide:

    Query: [________________________]

    [Separate]

Show:

    queued
    processing
    complete
    failed

After completion:

    [Play Isolated]
    [Save As...]

---

# 8. Suggested Query

When classification exists, show a suggested query.

Example:

    Classification:
        Footsteps

    Suggested query:
        "footsteps"

The user must be able to edit it.

---

# 9. Manual Selection

The waveform must support click-and-drag region selection.

Display:

    start
    end
    duration

Then:

    Query
    [Separate Selected Region]

---

# 10. Separation History

Display every separation attempt.

Example:

    Separation 001
    Query: "a whisper"
    Status: Complete
    Duration: 12.4 s

    Separation 002
    Query: "a person speaking"
    Status: Complete
    Duration: 11.8 s

Do not hide previous attempts.

---

# 11. System Status

Show:

    Capture: RUNNING
    Analysis: RUNNING
    Separation: IDLE

And:

    CPU
    RAM
    Queue depth
    Separation processing time

---

# 12. Error Display

Errors must be visible and understandable.

Example:

    Separation failed:
    model execution returned an error.

Do not silently swallow failures.

---

# 13. CPU Overload

If the system is overloaded, show the condition.

Example:

    Separation delayed — live capture has priority.

The application must continue capturing audio.
