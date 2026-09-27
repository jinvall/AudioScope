# Phase 01 — Audio Foundation

## Objective

Build the reliable audio foundation before implementing detection or machine learning.

---

## Tasks

### 1. Device Discovery

Implement:

- list input devices
- identify default input
- select device by configuration

---

### 2. Capture

Implement:

- continuous microphone capture
- internal float32 representation
- preferred 48 kHz
- mono processing path

---

### 3. Ring Buffer

Implement configurable rolling audio storage.

Default:

    30 seconds

---

### 4. Playback

Implement:

- play
- pause
- stop
- seek
- volume

Playback must use actual audio.

---

### 5. Recording

Implement event-independent raw recording.

Use WAV.

---

## Acceptance

Demonstrate:

    microphone
       ↓
    capture
       ↓
    ring buffer
       ↓
    recording
       ↓
    playback

No detector or separator is required yet.
