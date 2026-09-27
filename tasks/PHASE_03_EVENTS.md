# Phase 03 — Event Detection and Classification

## Objective

Turn continuous acoustic measurements into persistent events.

## Tasks

Implement:

- candidate detector
- event state machine
- onset detection
- release detection
- pre-roll
- post-roll
- event merging
- event IDs
- event metadata

Implement classification support for:

- whisper
- speech
- footsteps
- movement
- interference
- unknown

---

## Acceptance

A detected event must:

1. Have a timestamp.
2. Have a duration.
3. Contain actual captured audio.
4. Have a classification or unknown result.
5. Be stored persistently.
