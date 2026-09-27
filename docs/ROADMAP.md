# Development Roadmap

## Status

This roadmap's phase numbering is its own; it does not match the phase task
files in `tasks/`, which split the work differently. Where they differ, the
`tasks/` files are what was actually planned against.

    Phase 1  Audio Foundation      complete
    Phase 2  Analysis              complete
    Phase 3  Event Engine          complete
    Phase 4  Classification       complete
    Phase 5  Source Separation    not started
    Phase 6  GUI                   done. PyQt5 review workstation over the
                                    existing EventDatabase, styled from the
                                    staged theme pack in
                                    app/gui/srp-css-theme-pack/
    Phase 7  Storage               complete for events and recordings;
                                    the audio retention policy is not implemented
    Phase 8  Testing               352 tests passing, stable across repeated runs

Implemented behaviour and the measurements behind it are documented in
`docs/ANALYSIS.md`, `docs/DETECTION_IMPLEMENTATION.md`,
`docs/EVENT_DATABASE.md` and `docs/GUI.md`.

Note the phase numbering here differs from `tasks/`, which splits the work
differently; the Phase 5 GUI corresponds to this roadmap's Phase 6.

---

## Phase 1 — Audio Foundation

Implement:

- audio device discovery
- live capture
- format conversion
- ring buffer
- playback
- basic recording

Completion:

    live microphone audio can be captured and replayed.

---

# Phase 2 — Analysis

Implement:

- FFT
- waveform
- spectrogram
- RMS
- peak
- spectral features
- band energy
- adaptive noise floor

Completion:

    live audio produces useful measurable features.

---

# Phase 3 — Event Engine

Implement:

- candidate detection
- event state machine
- pre-roll
- post-roll
- event storage
- event timeline

Completion:

    acoustic events become persistent event objects.

---

# Phase 4 — Classification

Implement:

- whisper analysis
- footstep analysis
- movement analysis
- interference analysis
- unknown handling

Completion:

    events have useful classifications or remain unknown.

---

# Phase 5 — Source Separation

Implement:

- real model integration
- query input
- separation manager
- asynchronous queue
- output validation
- raw isolated WAV
- optional enhancement

Completion:

    a selected event can produce a real isolated audio file.

---

# Phase 6 — GUI

Implement:

- event list
- event detail
- waveform selection
- playback
- separation controls
- result history
- system status

Completion:

    the application is usable without CLI interaction.

---

# Phase 7 — Storage

Implement:

- event directories
- metadata
- continuous recordings
- separation history
- retention configuration

---

# Phase 8 — Testing

Implement:

- unit tests
- synthetic audio tests
- integration tests
- separation tests
- CPU tests
- long-running tests

---

# Final Completion Criteria

Source separation and the GUI are the two remaining gaps. Until both exist the
application cannot do the central thing it exists for: take an event, isolate
the queried sound from it, and let a person listen to the result.

The application is complete when:

    live audio
        ↓
    event detection
        ↓
    event selection
        ↓
    separation query
        ↓
    actual source separation
        ↓
    isolated audio
        ↓
    playback
        ↓
    save

works reliably on CPU-only Linux hardware.
