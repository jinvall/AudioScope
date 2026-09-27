# Phase 06 — Integration

## Objective

Connect every subsystem into one working application.

## Required Pipeline

    LIVE AUDIO
        ↓
    RING BUFFER
        ↓
    ANALYSIS
        ↓
    EVENT DETECTION
        ↓
    CLASSIFICATION
        ↓
    EVENT STORAGE
        ↓
    SEPARATION
        ↓
    ISOLATED OUTPUT
        ↓
    PLAYBACK
        ↓
    SAVE

---

## Integration Tests

Test:

- whisper
- footsteps
- movement
- interference
- unknown
- manual region

---

## CPU Test

Run entirely without GPU.

---

## Long Runtime

Run continuously and monitor:

- CPU
- RAM
- queues
- dropped audio
- recording integrity
- GUI responsiveness

---

## Failure Test

Cause source separation to fail.

Verify:

- original remains
- GUI remains responsive
- failure is reported
- another attempt is possible

---

## Final Acceptance

A user can hear a sound in the mixed live stream, identify/select the event, request isolation, and actually listen to the resulting isolated audio.

That is the final definition of functional success.
